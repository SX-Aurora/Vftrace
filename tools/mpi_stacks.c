/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <byteswap.h>
#include "vftr_scenarios.h"
#include "vftr_filewrite.h"
#include "vftr_mpi_utils.h"

#define RECORD_LENGTH 10240


/* FIXME make common header file */

typedef struct FunctionEntry {
    char  *name;
    double elapse_time;
} function_entry_t;

typedef struct stack_leaf {
	char *function_name;
	char *module_name;
	struct stack_leaf *next_in_level;
	struct stack_leaf *callee;	
	struct stack_leaf *origin;
	double entry_time;
	double time_spent;
} stack_leaf_t;	
	

static int
cmpstring( const void *p1, const void *p2 )
{
    function_entry_t *e1 = ( function_entry_t * ) p1;
    function_entry_t *e2 = ( function_entry_t * ) p2;

    return strcmp( e1->name, e2->name );
}

typedef struct FileHeader {
    char         fileid[VFTR_FILEIDSIZE], date[24];
    long long    interval;
    int          threads, thread, tasks, task; 
    union { double d; unsigned long long l; } cycletime, runtime;
    long long inittime;
    unsigned int samplecount, sampleoffset;
    unsigned int stackscount, stacksoffset;
    unsigned int reserved;
    int n_perf_types;
} vfdhdr_t;

void read_fileheader (vfdhdr_t *vfdhdr, FILE *fp);
void print_fileheader (vfdhdr_t vfdhdr);
void print_stacktree (stack_leaf_t *leaf, int n_spaces, double *total_mpi_time);


char *strip_trailing_asterisk (char *s) {
	int n = strlen(s);
	if (s[n-1] == '*') {
		s[n-1] = 0;
	}
	return strdup(s);
}

typedef struct StackEntry {
    char  *name;
    int    caller;
    double entry_time;
    int    fun;
} stack_entry_t;

void fill_into_stack_tree (stack_leaf_t **this_leaf, stack_entry_t *stacks,
			   int stackID_0, int sidw, double stime) {
  	int stackID = stackID_0;
	int stack_ids[100];
	int n_stack_ids = 0;
	for ( ;; stackID = stacks[stackID].caller) {
		stack_ids[n_stack_ids++] = stackID;
		if (stackID == 0) break;
	}
	stackID = stack_ids[n_stack_ids - 1];
	if (*this_leaf) {
		*this_leaf = (*this_leaf)->origin;
	} else {
		*this_leaf = (stack_leaf_t*) malloc (sizeof(stack_leaf_t));
		(*this_leaf)->function_name = strdup(stacks[stackID].name);
		(*this_leaf)->module_name = "";
		(*this_leaf)->next_in_level = NULL;
		(*this_leaf)->callee = NULL;
		(*this_leaf)->origin = (stack_leaf_t*) malloc (sizeof(stack_leaf_t));
		(*this_leaf)->origin = *this_leaf;
		(*this_leaf)->entry_time = 0.0;
		(*this_leaf)->time_spent = 0.0;
	}
	for (int level = n_stack_ids - 2; level >= 0; level--) {
		stackID = stack_ids[level];
		if ((*this_leaf)->callee) {
			*this_leaf = (*this_leaf)->callee;
			while (strcmp ((*this_leaf)->function_name, stacks[stackID].name)) {
				if ((*this_leaf)->next_in_level) {
					*this_leaf = (*this_leaf)->next_in_level;
				} else {
					(*this_leaf)->next_in_level = 
						(stack_leaf_t*) malloc (sizeof(stack_leaf_t));
					(*this_leaf)->next_in_level->function_name = strdup(stacks[stackID].name);
					(*this_leaf)->next_in_level->module_name = "";
					(*this_leaf)->next_in_level->next_in_level = NULL;	
					(*this_leaf)->next_in_level->callee = NULL;
					(*this_leaf)->next_in_level->origin = (stack_leaf_t*)malloc (sizeof(stack_leaf_t));
					(*this_leaf)->next_in_level->origin = (*this_leaf)->origin;
					if (level == 0) {
						if (sidw == SID_ENTRY) {
							(*this_leaf)->next_in_level->entry_time = stime;
						} else {
							(*this_leaf)->next_in_level->time_spent += (stime - (*this_leaf)->next_in_level->entry_time);
						}
					} else {
						(*this_leaf)->next_in_level->entry_time = 0.0;
						(*this_leaf)->next_in_level->time_spent = 0.0;
					}
					*this_leaf = (*this_leaf)->next_in_level;
					break;
				}
			}
			if (level == 0) {
				if (sidw == SID_ENTRY) {
					(*this_leaf)->entry_time = stime;
				} else {
					(*this_leaf)->time_spent += (stime - (*this_leaf)->entry_time);
				}
			}	
		} else {
			(*this_leaf)->callee = (stack_leaf_t*) malloc (sizeof(stack_leaf_t));
			(*this_leaf)->callee->function_name = strdup(stacks[stackID].name);	
			(*this_leaf)->callee->module_name = "";
			(*this_leaf)->callee->next_in_level = NULL;
			(*this_leaf)->callee->callee = NULL;
			(*this_leaf)->callee->origin = (stack_leaf_t*)malloc (sizeof(stack_leaf_t));
			(*this_leaf)->callee->origin = (*this_leaf)->origin;
			if (level == 0) {
				if (sidw == SID_ENTRY) {
					(*this_leaf)->callee->entry_time = stime;	
				} else {
					(*this_leaf)->callee->time_spent += (stime - (*this_leaf)->callee->entry_time);
				}
			} else {
				(*this_leaf)->callee->entry_time = 0.0;
				(*this_leaf)->callee->time_spent = 0.0;
			}
			*this_leaf = (*this_leaf)->callee;
		}
	}	
}

int main (int argc, char **argv) {
    FILE      *fp;
    long long  time0;
    char       record[RECORD_LENGTH], *s;
    char      *typename[34];
    int        i, j, samples, nextID, levels, caller, nb_functions, show_precise; 
    long file_size, max_fp;
    double     dtime = 0.;
    char *filename, *search_func;
    function_entry_t *functions = NULL;
    vfdhdr_t vfdhdr;

    stack_entry_t *stacks;

    struct StackInfo {
        int id, levels, caller, len;
    } stackInfo;
    

    union { unsigned int       ui[2];
            unsigned long long ull;   } sid;

    int this_vfd_version;

    nb_functions   = 0;
    
    if (argc < 3) {
	    printf ("Usage: tracedump <vfd-file> <search_func>\n");
	    return -1;
    }

    filename = argv[1];
    search_func = argv[2];
    show_precise = 0;

    fp = fopen( filename, "r" );
    assert( fp );
    fseek (fp, 0L, SEEK_END);
    file_size = ftell(fp);
    max_fp = 0;
    rewind(fp);

    printf ("Reading: %s; Size: %ld bytes\n", filename, file_size);

    fread (&this_vfd_version, 1, sizeof(int), fp);
    printf ("VFD version: %d\n", this_vfd_version);
    if (this_vfd_version != VFD_VERSION) {
	printf ("The file %s does not have the most recent VFD version (%d)!\n",
		filename, VFD_VERSION);  
	return -1;
    }
    read_fileheader (&vfdhdr, fp);

    printf ("header size=%ld offset=%ld\n",
	     sizeof(struct FileHeader), ftell(fp));
    print_fileheader (vfdhdr);

    fread (&(vfdhdr.n_perf_types), sizeof(int), 1, fp);
    double *perf_values = NULL;
    if (vfdhdr.n_perf_types > 0) {
    	char scenario_name[SCENARIO_NAME_LEN];
	perf_values = (double*)malloc (vfdhdr.n_perf_types * sizeof(double));
        for (i = 0; i < vfdhdr.n_perf_types; i++) {
            	fread (scenario_name, SCENARIO_NAME_LEN, 1, fp);
            	printf ("Performance counter name: %s\n", scenario_name);
            	int perf_integrated;
            	fread (&perf_integrated, sizeof(int), 1, fp);
            	printf ("Integrated counter: ");
            	perf_integrated == 0 ? printf ("NO\n") : printf ("YES\n");
	        perf_values[i] = 0.0;
        }
    }
    
    printf ("Unique stacks:   %d\n",                 vfdhdr.stackscount);
    if (!show_precise) printf ("Stacks list:\n");
    stacks = (struct StackEntry *) malloc (vfdhdr.stackscount * sizeof(struct StackEntry));
    if (ftell(fp) > max_fp) max_fp = ftell(fp);
    fseek (fp, vfdhdr.stacksoffset, SEEK_SET);
    
    for (i = 0; i < vfdhdr.stackscount; i++) {
	fread (&stackInfo, sizeof(int), 4, fp);
        int len = stackInfo.len < RECORD_LENGTH ? stackInfo.len : RECORD_LENGTH - 1;
	fread (record, sizeof(char), len, fp);
	record[len] = 0;
        if(!show_precise) {
          printf ("      %d,%d,%d,%s\n",
                   stackInfo.id, stackInfo.levels, stackInfo.caller, record);
	}
        stacks[stackInfo.id].name   = strip_trailing_asterisk(record);
	stacks[stackInfo.id].caller = stackInfo.caller;

        stacks[stackInfo.id].fun = -1;
        if( record[strlen(record) - 1] == '*' ) {
            for( j = 0; j < nb_functions; j++ ) {
                if( !strcmp( record, functions[j].name ) ) {
                    stacks[stackInfo.id].fun = j;
                    break;
                }
            }
            if( stacks[stackInfo.id].fun == -1 ) {
                nb_functions++;
                if ( nb_functions == 1 ) {
                    functions = (struct FunctionEntry *) malloc( nb_functions * sizeof(struct FunctionEntry) );
                } else {
                    functions = (struct FunctionEntry *) realloc( functions, nb_functions * sizeof(struct FunctionEntry) );
                }

                functions[nb_functions - 1].name = strip_trailing_asterisk (record);
                functions[nb_functions - 1].elapse_time = 0.0;
            }
        }

        for( j = 0; j < nb_functions; j++ ) {
            if( !strcmp( stacks[stackInfo.id].name, functions[j].name ) ) {
                stacks[stackInfo.id].fun = j;
                break;
            }
        }
    }
    
    if (ftell(fp) > max_fp) max_fp = ftell(fp);
    fseek( fp, vfdhdr.sampleoffset, SEEK_SET );

    if( !show_precise )printf( "\nStack and message samples:\n\n" );
  
    unsigned long long hashes [vfdhdr.samplecount];
    stack_leaf_t *stack_tree = NULL;
    stack_leaf_t *this_leaf = stack_tree;

    for(i = 0; i < vfdhdr.samplecount; i++ ) {
        int        sidw;
	long       pos;

	hashes[i] = 0;
        fread (&sidw, sizeof(int), 1, fp);

        if (sidw == SID_MESSAGE) {
            int dir, type_idx, rank; 
            int type_size;
            long long tstart, tstop;
	    unsigned int count, tag;
	    fread (&dir, sizeof(int), 1, fp);
            fread (&rank, sizeof(int), 1, fp);
            fread (&type_idx, sizeof(int), 1, fp);
            fread (&count, sizeof(int), 1, fp);
            fread (&type_size, sizeof(int), 1, fp);
	    fread (&tag, sizeof(int), 1, fp);
            fread (&tstart, sizeof(long long), 1, fp);
	    fread (&tstop, sizeof(long long), 1, fp);
            double dtstart = tstart * 1.0e-6;
            double dtstop = tstop * 1.0e-6;
            double rate = count*type_size/(dtstop-dtstart)/(1024.0*1024.0);
        } else if (sidw == SID_ENTRY || sidw == SID_EXIT) {
            int stackID;
            fread (&stackID, sizeof(int), 1, fp);
            long long ltime = 0;
            fread (&ltime, sizeof (long long), 1, fp);
            double stime = ltime * 1.0e-6;
	    for (int p = 0; p < vfdhdr.n_perf_types; p++) {
		fread (&perf_values[p], sizeof(double), 1, fp);
	    }

	    if (!strcmp (stacks[stackID].name, search_func)) {
		fill_into_stack_tree(&this_leaf, stacks, stackID, sidw, stime);
	    }
            if( stacks[stackID].fun != -1 ) {
                if (sidw == SID_ENTRY) {
			stacks[stackID].entry_time = stime;
                } else {
			functions[stacks[stackID].fun].elapse_time += (stime - stacks[stackID].entry_time);
                }
            }

	} else {
            printf("ERROR: Invalid sample type: %d\n", sidw);
            return 1;
        }
    }

    printf ("Check if leaf\n");
    if (!this_leaf) {
	printf ("No leaf!\n");
    } else {
    if (!this_leaf->origin) {
		printf ("Has no origin!\n");
    } else {
		printf ("Origin name: %s\n", this_leaf->origin->function_name);
    }}
    double total_mpi_time = 0.0;
    print_stacktree (this_leaf->origin, 0, &total_mpi_time);
    printf ("Total MPI time: %lf\n", total_mpi_time);

    if (!show_precise) printf ("\n");

    if (file_size != max_fp) {
	    printf ("WARNING: Not all data have been read!\n");
	    printf ("Currently at %ld, but the file has %ld bytes.\n",
		    max_fp, file_size);
    } else {
	    printf ("SUCCESS: All bytes have been read\n");
    }
    (void) fclose( fp );

    if(nb_functions != 0) {
        int l = 0;
        char fmt[16];

        qsort(functions, nb_functions, sizeof(function_entry_t), cmpstring);

        for( i = 0; i < nb_functions; i++ ) {
            if( strlen(functions[i].name) > l )
                l = strlen(functions[i].name);
        }
        sprintf( fmt, "%%%ds %%12.6f\n", l + 2 );
        fprintf( stdout, "\nElapse time for \"precise\" functions (including sub routine):\n\n" );
        for( i = 0; i < nb_functions; i++ ) {
            fprintf( stdout, fmt, functions[i].name, functions[i].elapse_time );
        }
    }

    if (perf_values) free (perf_values);
    free (stacks);
    free (functions);

    return 0;
}

void read_fileheader (vfdhdr_t *vfdhdr, FILE *fp) {
    fread( &vfdhdr->fileid,	 1, VFTR_FILEIDSIZE, 	  fp );
    fread( &vfdhdr->date,	 1, 24, 	          fp );
    fread( &vfdhdr->interval,	 1, sizeof(long long),    fp );
    fread( &vfdhdr->threads,	 1, sizeof(int),          fp );
    fread( &vfdhdr->thread,	 1, sizeof(int),          fp );
    fread( &vfdhdr->tasks,	 1, sizeof(int),          fp );
    fread( &vfdhdr->task,	 1, sizeof(int),          fp );
    fread( &vfdhdr->cycletime.l,  1, sizeof(long long),    fp );
    fread( &vfdhdr->inittime,    1, sizeof(long long),    fp );
    fread( &vfdhdr->runtime.l,	 1, sizeof(long long),    fp );
    fread( &vfdhdr->samplecount,  1, sizeof(unsigned int), fp );
    fread( &vfdhdr->stackscount,  1, sizeof(unsigned int), fp );
    fread( &vfdhdr->stacksoffset, 1, sizeof(unsigned int), fp );
    fread( &vfdhdr->sampleoffset, 1, sizeof(unsigned int), fp );
    fread( &vfdhdr->reserved,     1, sizeof(unsigned int), fp );
}

void print_fileheader (vfdhdr_t vfdhdr) {
    int i;
    // We require a seperate datestring which is one element larger than
    // the field in the vfd file to add a terminating null character. This
    // is not necessary for the version string since it is written using sprintf
    // in the main code.
    char       datestring[25], record[RECORD_LENGTH], *s;
    datestring[24] = 0;
    strncpy( datestring, vfdhdr.date,   24 );

    printf( "Version ID:      %s\n",		     vfdhdr.fileid    );
    printf( "Date:            %s\n",		     datestring );
    printf( "MPI tasks:       rank=%d count=%d\n",   vfdhdr.task,   vfdhdr.tasks   ); 
    printf( "OpenMP threads:  thread=%d count=%d\n", vfdhdr.thread, vfdhdr.threads );
    printf( "Sample interval: %12.6le seconds\n",    vfdhdr.interval*1.0e-6);
    printf( "Init time: %lld\n", vfdhdr.inittime);
    printf( "Job runtime:     %.3lf seconds\n",     vfdhdr.runtime.d    );
    printf( "Samples:         %d\n",                 vfdhdr.samplecount  );
    printf( "Unique stacks:   %d\n",                 vfdhdr.stackscount  );
    printf( "Stacks offset:   %d\n",                 vfdhdr.stacksoffset );
    printf( "Sample offset:   %d\n",                 vfdhdr.sampleoffset );
}

void print_stacktree (stack_leaf_t *leaf, int n_spaces, double *total_mpi_time) {
	if (!leaf) return;
	stack_leaf_t *new_leaf = (stack_leaf_t*)malloc (sizeof(stack_leaf_t));
	memcpy (new_leaf, leaf, sizeof(stack_leaf_t));
	//printf ("%s", leaf->function_name);
	//printf ("%s(%d)", new_leaf->function_name, n_spaces);
	printf ("%s", new_leaf->function_name);
	if (new_leaf->callee) {
		printf (">");
		//print_stacktree (leaf->callee, NULL);
		int new_n_spaces = n_spaces + strlen(new_leaf->function_name) + 1;
		print_stacktree (new_leaf->callee, new_n_spaces, total_mpi_time);
	} else {
		printf (": MPI time %4.3f s\n", new_leaf->time_spent);	
		*total_mpi_time = *total_mpi_time + new_leaf->time_spent;
	}
	if (new_leaf->next_in_level) {
		for (int i = 0; i < n_spaces; i++) printf (" ");
		printf (">");
		//print_stacktree (leaf->next_in_level, NULL);
		//int new_n_spaces = n_spaces + strlen(new_leaf->function_name) + 1;
		print_stacktree (new_leaf->next_in_level, n_spaces, total_mpi_time);
	}
	free (new_leaf);
}
