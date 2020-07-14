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


int
main( int argc, char **argv )
{
    FILE      *fp;
    long long  time0;
    char       record[RECORD_LENGTH], *s;
    char      *typename[34];
    int        i, j, samples, nextID, levels, caller, nb_functions, show_precise; 
    long file_size, max_fp;
    double     dtime = 0.;
    char *filename;
    function_entry_t *functions = NULL;
    vfdhdr_t vfdhdr;

    struct StackEntry {
        char  *name;
	int    caller;
        double entry_time;
        int    fun;
    } *stacks;

    struct StackInfo {
        int id, levels, caller, len;
    } stackInfo;
    

    union { unsigned int       ui[2];
            unsigned long long ull;   } sid;

    int this_vfd_version;

    nb_functions   = 0;
    
    if (argc < 2) {
	    printf ("Usage: tracedump <vfd-file>\n");
	    return -1;
    } else if (argc > 2 && !strcmp (argv[1], "-p")) {
	    	printf ("Attention: More than one input file. Only the first is processed\n");
    } else if (argc > 3 && strcmp (argv[1], "-p")) {
	    	printf ("Attention: More than one input file. Only the first is processed\n");
    }

    if( !strcmp( argv[1], "-p" ) ) {
        show_precise = 1;
	filename = argv[2];
    } else {
        show_precise = 0;
	filename = argv[1];
    }
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

    printf( "hdrsize=%ld offset=%ld\n",
	     sizeof(struct FileHeader), ftell(fp));
    print_fileheader (vfdhdr);

    fread (&(vfdhdr.n_perf_types), sizeof(int), 1, fp);
    printf ("n_perf_types: %d\n", vfdhdr.n_perf_types);
    char name[SCENARIO_NAME_LEN];
    double *perf_values;
    if (vfdhdr.n_perf_types > 0) {
	perf_values = (double*)malloc (vfdhdr.n_perf_types * sizeof(double));
        for (i = 0; i < vfdhdr.n_perf_types; i++) {
            	fread (name, SCENARIO_NAME_LEN, 1, fp);
            	printf ("Performance counter name: %s\n", name);
            	int perf_integrated;
            	fread (&perf_integrated, sizeof(int), 1, fp);
            	printf ("Integrated counter: ");
            	perf_integrated == 0 ? printf ("NO\n") : printf ("YES\n");
	        perf_values[i] = 0.0;
        }
    }
    
    printf( "Unique stacks:   %d\n",                 vfdhdr.stackscount  );
    if( !show_precise )printf( "Stacks list:\n" );
    stacks = (struct StackEntry *) malloc( vfdhdr.stackscount * 
                                              sizeof(struct StackEntry) );    
    if (ftell(fp) > max_fp) max_fp = ftell(fp);
    fseek( fp, vfdhdr.stacksoffset, SEEK_SET );
    
    for( i=0; i<vfdhdr.stackscount; i++ ) {
	fread( &stackInfo, sizeof(int), 4, fp );
        int len = stackInfo.len < RECORD_LENGTH ? stackInfo.len : RECORD_LENGTH - 1;
	fread( record, sizeof(char), len, fp );
	record[len] = 0;
	if( !show_precise )printf( "                 %d,%d,%d,%s\n", 
	        stackInfo.id, stackInfo.levels, stackInfo.caller, record );
        stacks[stackInfo.id].name   = strdup( record );
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

                functions[nb_functions - 1].name = strdup( record );
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

    for(i = 0; i < vfdhdr.samplecount; i++ ) {
        int        sidw;
	long       pos;

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
            printf("%16.6f %s\n", dtstart, dir ? "recv" : "send");
            printf("%16s count=%d type=%s(%iBytes) rate= %8.4lf MiB/s peer=%d tag=%d\n",
                   "",
		   count,
                   vftr_get_mpitype_string_from_idx(type_idx), 
                   type_size,
                   rate, 
                   rank,
                   tag);
            printf("%16.6f %s end\n", dtstop, dir ? "recv" : "send");
        } else if (sidw == SID_ENTRY || sidw == SID_EXIT) {
            int stackID;
            fread (&stackID, sizeof(int), 1, fp);
            long long ltime = 0;
            fread (&ltime, sizeof (long long), 1, fp);
            double stime = ltime * 1.0e-6;
	    for (int p = 0; p < vfdhdr.n_perf_types; p++) {
		fread (&perf_values[p], sizeof(double), 1, fp);
	    }

            if( stacks[stackID].fun != -1 ) {
                if (sidw == SID_ENTRY) {
			stacks[stackID].entry_time = stime;
                } else {
			functions[stacks[stackID].fun].elapse_time += (stime - stacks[stackID].entry_time);
                }
            }

            if(!show_precise) {
		printf("%16.6f %s ", stime, sidw == SID_ENTRY ? "call" : "exit");
		for( ;; stackID = stacks[stackID].caller ) {
		    printf( "%s%s", stacks[stackID].name, stackID ? "<" : "" );
  		    if(stackID == 0) break;
                }
	        printf("\n");
            }
	} else {
            printf("ERROR: Invalid sample type: %d\n", sidw);
            return 1;
        }
    }

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

    free (perf_values);
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
