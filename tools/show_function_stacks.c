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

#include "vftr_vfd_utils.h"

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

void print_stacktree (stack_leaf_t *leaf, int n_spaces, double *total_mpi_time);

bool is_precise (char *s) {
 	return s[strlen(s)-1] == '*';
}

char *strip_trailing_asterisk (char *s) {
	char *sdup = strdup(s);
	int n = strlen(sdup);
	if (sdup[n-1] == '*') {
		sdup[n-1] = '\0';
	}
	return sdup;
}

typedef struct StackEntry {
    char *name;
    int caller;
    double entry_time;
    int fun;
    bool precise;
} stack_entry_t;

void fill_into_stack_tree (stack_leaf_t **this_leaf, stack_entry_t *stacks,
			   int stackID_0, int sample_id, double stime) {
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
						if (sample_id == SID_ENTRY) {
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
				if (sample_id == SID_ENTRY) {
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
				if (sample_id == SID_ENTRY) {
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

void read_stacks (FILE *fp, stack_entry_t **stacks, function_entry_t **functions, 
		  unsigned int stacks_count,
                  unsigned int stacks_offset, int *n_precise_functions, long *max_fp) {

    struct StackInfo {
        int id, levels, caller, len;
    } stackInfo;
    char record[RECORD_LENGTH];

    *stacks = (stack_entry_t *) malloc (stacks_count * sizeof(stack_entry_t));
    if (max_fp && ftell(fp) > *max_fp) {
       *max_fp = ftell(fp);
    }
    fseek (fp, stacks_offset, SEEK_SET);
    
    for (int i = 0; i < stacks_count; i++) {
	fread (&stackInfo, sizeof(int), 4, fp);
        int len = stackInfo.len < RECORD_LENGTH ? stackInfo.len : RECORD_LENGTH - 1;
	fread (record, sizeof(char), len, fp);
	record[len] = '\0';
        (*stacks)[stackInfo.id].name = strip_trailing_asterisk(record);
	(*stacks)[stackInfo.id].caller = stackInfo.caller;
	(*stacks)[stackInfo.id].precise = is_precise(record);

        (*stacks)[stackInfo.id].fun = -1;
        if ((*stacks)[stackInfo.id].precise) {
            for (int j = 0; j < *n_precise_functions; j++) {
                if (!strcmp (record, (*functions)[j].name)) {
                    (*stacks)[stackInfo.id].fun = j;
                    break;
                }
            }
	    
            if ((*stacks)[stackInfo.id].fun == -1 ) {
                (*n_precise_functions)++;
                if  (*n_precise_functions == 1) {
                    *functions = (struct FunctionEntry *) malloc (*n_precise_functions * sizeof(struct FunctionEntry) );
                } else {
                    *functions = (struct FunctionEntry *) realloc (*functions, *n_precise_functions * sizeof(struct FunctionEntry) );
                }

                (*functions)[*n_precise_functions - 1].name = strip_trailing_asterisk (record);
                (*functions)[*n_precise_functions - 1].elapse_time = 0.0;
            }
        }

        for (int j = 0; j < *n_precise_functions; j++) {
            if (!strcmp ((*stacks)[stackInfo.id].name, (*functions)[j].name)) {
                (*stacks)[stackInfo.id].fun = j;
                break;
            }
        }
    }
}


int main (int argc, char **argv) {
    FILE *fp;
    int n_precise_functions; 
    char *filename, *search_func;

    vfd_header_t vfd_header;
    function_entry_t *functions = NULL;
    stack_entry_t *stacks = NULL;

    n_precise_functions = 0;
    
    if (argc < 3) {
	    printf ("Usage: show_function_stacks <vfd-file> <search_func>\n");
	    return -1;
    }

    filename = argv[1];
    search_func = argv[2];

    fp = fopen (filename, "r");
    assert (fp);

    printf ("Reading: %s Analyzing: %s\n", filename, search_func);

    // We are not interested in the VFD version here
    int dummy;
    fread (&dummy, 1, sizeof(int), fp);
    // From the header, we actually only need the stack and sample offset
    read_fileheader (&vfd_header, fp);

    printf ("header size = %ld offset = %ld\n",
	     sizeof(struct FileHeader), ftell(fp));

    // We need the number of hardware scenarios, because when scanning the samples
    // and a message is encountered (sample_id == SID_MESSAGE), we need to scan over these
    // values in order to be synchronized. Also, we allocate the corresponding (dummy-)buffer
    fread (&(vfd_header.n_perf_types), sizeof(int), 1, fp);
    
    printf ("Unique stacks:   %d\n", vfd_header.stackscount);
    read_stacks (fp, &stacks, &functions,
		 vfd_header.stackscount, vfd_header.stacksoffset, 
                 &n_precise_functions, NULL);
    
    fseek (fp, vfd_header.sampleoffset, SEEK_SET);

    stack_leaf_t *stack_tree = NULL;
    bool has_been_warned = false;

    for (int i = 0; i < vfd_header.samplecount; i++ ) {
        int sample_id;

        fread (&sample_id, sizeof(int), 1, fp);

        if (sample_id == SID_MESSAGE) {
	    fseek (fp, ftell(fp) + 6 * sizeof(int) + 2 * sizeof(long long), SEEK_SET);
        } else if (sample_id == SID_ENTRY || sample_id == SID_EXIT) {
            int stackID;
            fread (&stackID, sizeof(int), 1, fp);
            long long ltime = 0;
            fread (&ltime, sizeof (long long), 1, fp);
            double stime = ltime * 1.0e-6;
	    fseek (fp, ftell(fp) + vfd_header.n_perf_types * sizeof(double), SEEK_SET);

	    if (!strcmp (stacks[stackID].name, search_func)) {
		if ((!stacks[stackID].precise) && (!has_been_warned)) {
			printf ("Attention: The function %s is not precise. \n"
				"The data printed here is unreliable. "
				"Please sample again using VFTR_PRECISE.\n",
				stacks[stackID].name);
			has_been_warned = true;
		}
		fill_into_stack_tree(&stack_tree, stacks, stackID, sample_id, stime);
	    }
	} else {
            printf("ERROR: Invalid sample type: %d\n", sample_id);
            return 1;
        }
    }

    double total_mpi_time = 0.0;
    print_stacktree (stack_tree->origin, 0, &total_mpi_time);
    printf ("Total MPI time: %lf\n", total_mpi_time);

    fclose (fp);

    free (stacks);
    free (functions);

    return 0;
}

void print_stacktree (stack_leaf_t *leaf, int n_spaces, double *total_mpi_time) {
	if (!leaf) return;
	printf ("%s", leaf->function_name);
	if (leaf->callee) {
		printf (">");
		int new_n_spaces = n_spaces + strlen(leaf->function_name) + 1;
		print_stacktree (leaf->callee, new_n_spaces, total_mpi_time);
	} else {
		printf (": MPI time %4.3f s\n", leaf->time_spent);	
		*total_mpi_time = *total_mpi_time + leaf->time_spent;
	}
	if (leaf->next_in_level) {
		for (int i = 0; i < n_spaces; i++) printf (" ");
		printf (">");
		print_stacktree (leaf->next_in_level, n_spaces, total_mpi_time);
	}
}
