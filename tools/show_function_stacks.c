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
#include <limits.h>
#include "vftr_scenarios.h"
#include "vftr_filewrite.h"
#include "vftr_mpi_utils.h"
#include "vftr_sorting.h"

#include "vftr_vfd_utils.h"

typedef struct stack_leaf {
	char *function_name;
	char *module_name;
	struct stack_leaf *next_in_level;
	struct stack_leaf *callee;	
	struct stack_leaf *origin;
	double *entry_time;
	double *time_spent;
} stack_leaf_t;	

static int n_vfds; // Number of vfd files which are read in

/**********************************************************************/

void evaluate_mpi_time (double *all_times, 
			double *t_avg, double *t_min, double *t_max, double *imbalance) {
	*t_avg = 0.0;
	*t_min = LONG_MAX;
	*t_max = 0.0; 
	int n = 0;
	for (int i = 0; i < n_vfds; i++) {
		if (all_times[i] > 0) {
			n++;
			*t_avg = ((n - 1) * (*t_avg) + all_times[i]) / n;

			if (all_times[i] > *t_max) {
				*t_max = all_times[i];
			}
			if (all_times[i] < *t_min) {
				*t_min = all_times[i];
			}
		}
	}
	double d1 = *t_avg - *t_min;
	double d2 = *t_max - *t_avg;
	*imbalance = d1 > d2 ? d1 / *t_avg * 100 : d2 / *t_avg * 100;
}

/**********************************************************************/

void print_stacktree (stack_leaf_t *leaf, int n_spaces, double *total_mpi_time) {
	if (!leaf) return;
	printf ("%s", leaf->function_name);
	if (leaf->callee) {
		printf (">");
		int new_n_spaces = n_spaces + strlen(leaf->function_name) + 1;
		print_stacktree (leaf->callee, new_n_spaces, total_mpi_time);
	} else {
		double t_avg, t_min, t_max, imbalance;
		evaluate_mpi_time (leaf->time_spent,
				   &t_avg, &t_min, &t_max, &imbalance);
		printf (": MPI %4.3f %4.3f %4.3f %4.2f %%\n", t_avg, t_min, t_max, imbalance);
		*total_mpi_time = *total_mpi_time + leaf->time_spent[0];
	}
	if (leaf->next_in_level) {
		for (int i = 0; i < n_spaces; i++) printf (" ");
		printf (">");
		print_stacktree (leaf->next_in_level, n_spaces, total_mpi_time);
	}
}

/**********************************************************************/

int count_stacks (stack_leaf_t *leaf, int *n_stacks) {
	if (!leaf) return;
	if (leaf->callee) {
		count_stacks (leaf->callee, n_stacks);
	} else {
		(*n_stacks)++;
	}
	if (leaf->next_in_level) {
		count_stacks (leaf->next_in_level, n_stacks);
	}
}

/**********************************************************************/

int get_hash (uint64_t this_hash, uint64_t *hashes, int n_hashes) {
	int i_hash = -1;
	//printf ("this hash: %llu\n", this_hash);
	for (int i = 0; i < n_hashes; i++) {
		//printf ("Compare: %llu\n", hashes[i]);
		if (hashes[i] == this_hash) {
			i_hash = i;
			break;
		}	
	}
	return i_hash;
}

/**********************************************************************/

void fill_into_stack_tree (stack_leaf_t **this_leaf, stack_entry_t *stacks,
			   int stackID_0, int sample_id, double stime, int i_vfd) {
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
		(*this_leaf)->entry_time = (double*)malloc (n_vfds * sizeof(double));
		(*this_leaf)->time_spent = (double*)malloc (n_vfds * sizeof(double));
		for (int i = 0; i < n_vfds; i++) {
			(*this_leaf)->entry_time[i] = 0.0;
			(*this_leaf)->time_spent[i] = 0.0;
		}
		//(*this_leaf)->entry_time = 0.0;
		//(*this_leaf)->time_spent = 0.0;
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
					(*this_leaf)->next_in_level->entry_time = (double*)malloc (n_vfds * sizeof(double));
					(*this_leaf)->next_in_level->time_spent = (double*)malloc (n_vfds * sizeof(double));						
					for (int i = 0; i < n_vfds; i++) {
							(*this_leaf)->next_in_level->entry_time[i_vfd] = 0.0;
							(*this_leaf)->next_in_level->time_spent[i_vfd] = 0.0;
					}

					if (level == 0) {
						if (sample_id == SID_ENTRY) {
							(*this_leaf)->next_in_level->entry_time[i_vfd] = stime;
						} else {
							(*this_leaf)->next_in_level->time_spent[i_vfd] += (stime - (*this_leaf)->next_in_level->entry_time[i_vfd]);
						}
					} else {
						//(*this_leaf)->next_in_level->entry_time = 0.0;
						//(*this_leaf)->next_in_level->time_spent = 0.0;
					}
					*this_leaf = (*this_leaf)->next_in_level;
					break;
				}
			}
			if (level == 0) {
				if (sample_id == SID_ENTRY) {
					(*this_leaf)->entry_time[i_vfd] = stime;
				} else {
					(*this_leaf)->time_spent[i_vfd] += (stime - (*this_leaf)->entry_time[i_vfd]);
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
				
			(*this_leaf)->callee->entry_time = (double*)malloc (n_vfds * sizeof(double));	
			(*this_leaf)->callee->time_spent = (double*)malloc (n_vfds * sizeof(double));	
			for (int i = 0; i < n_vfds; i++) {
					(*this_leaf)->callee->entry_time[i_vfd] = 0.0;
					(*this_leaf)->callee->time_spent[i_vfd] = 0.0;
			}

			if (level == 0) {
				if (sample_id == SID_ENTRY) {
					(*this_leaf)->callee->entry_time[i_vfd] = stime;	
				} else {
					(*this_leaf)->callee->time_spent[i_vfd] += (stime - (*this_leaf)->callee->entry_time[i_vfd]);
				}
			} else {
				//(*this_leaf)->callee->entry_time = 0.0;
				//(*this_leaf)->callee->time_spent = 0.0;
			}
			*this_leaf = (*this_leaf)->callee;
		}
	}	
}

/**********************************************************************/

int main (int argc, char **argv) {
    FILE *fp;
    int n_precise_functions; 
    char *filename, *search_func;

    vfd_header_t vfd_header;
    function_entry_t *precise_functions = NULL;
    stack_entry_t *stacks = NULL;
	
#define N_HASH_START 5000
#define HASH_INCREMENT 500
    int hashlist_size = N_HASH_START;
    uint64_t *hashes = (uint64_t*)malloc(hashlist_size * sizeof(uint64_t));;

    
    if (argc < 3) {
	    printf ("Usage: show_function_stacks <vfd-file> <search_func>\n");
	    return -1;
    }

    //filename = argv[1];
    //search_func = argv[2];
    n_vfds = argc - 2;
    search_func = argv[n_vfds + 1];
    int n_tot_stacks = 0;
    int i_hash = 0;
    bool vfd_has_no_stacks[n_vfds];
    for (int i = 0; i < n_vfds; i++) {
	vfd_has_no_stacks[i] = false;
    }

    bool *all_stack_ids[n_vfds];
    int n_stack_ids[n_vfds];
    
    stack_leaf_t *stack_tree = NULL;

    for (int i_vfd = 0; i_vfd < n_vfds; i_vfd++) {

	    filename = argv[i_vfd+1];
	    fp = fopen (filename, "r");
	    assert (fp);
	
	    //printf ("Reading: %s Analyzing: %s\n", filename, search_func);
	
	    // We are not interested in the VFD version here
	    int dummy;
	    fread (&dummy, 1, sizeof(int), fp);
	    // From the header, we actually only need the stack and sample offset
	    read_fileheader (&vfd_header, fp);
	
	    //printf ("header size = %ld offset = %ld\n",
//		     sizeof(struct FileHeader), ftell(fp));
	
	    // We need the number of hardware scenarios, because when scanning the samples
	    // and a message is encountered (sample_id == SID_MESSAGE), we need to scan over these
	    // values in order to be synchronized. Also, we allocate the corresponding (dummy-)buffer
	    fread (&(vfd_header.n_hw_obs), sizeof(int), 1, fp);
	    skip_hw_observables (fp, vfd_header.n_hw_obs);
	    //printf ("Unique stacks:   %d\n", vfd_header.stackscount);
	    
	    // Although not needed elsewhere here, we need the "precise_functions" array
	    // because it is used inside of read_stacks to compute indices. Other routines
	    // such as tracedump need it, so we keep it as an external field.
	    n_precise_functions = 0;
	    read_stacks (fp, &stacks, &precise_functions,
			 vfd_header.stackscount, vfd_header.stacksoffset, 
	                 &n_precise_functions, NULL);
	
	    for (int i = 0; i < vfd_header.stackscount; i++) {
		if (stacks[i].precise) {
			stacks[i].name = strip_trailing_asterisk(stacks[i].name);
		}
	    }
	
	    fseek (fp, vfd_header.sampleoffset, SEEK_SET);
	
	    bool has_been_warned = false;
	
	    all_stack_ids[i_vfd] = (bool*)malloc (vfd_header.stackscount * sizeof(bool));
	    n_stack_ids[i_vfd] = 0;
	    for (int i = 0; i < vfd_header.stackscount; i++) {
		all_stack_ids[i_vfd][i] = false;
	    }
	
	    for (int i = 0; i < vfd_header.samplecount; i++ ) {
	        int sample_id;
	
	        fread (&sample_id, sizeof(int), 1, fp);
	
	        if (sample_id == SID_MESSAGE) {
		    skip_mpi_message_sample (fp);
	        } else if (sample_id == SID_ENTRY || sample_id == SID_EXIT) {
	            int stack_id;
		    long long sample_time;
		    read_stack_sample (fp, vfd_header.n_hw_obs, &stack_id, &sample_time, NULL);
		    double sample_time_s = (double)sample_time * 1e-6;
	
		    if (!strcmp (stacks[stack_id].name, search_func)) {
			if ((!stacks[stack_id].precise) && (!has_been_warned)) {
				printf ("Attention: The function %s is not precise. \n"
					"The data printed here is unreliable. "
					"Please sample again using VFTR_PRECISE.\n",
					stacks[stack_id].name);
				has_been_warned = true;
			}
			fill_into_stack_tree (&stack_tree, stacks, stack_id,
					      sample_id, sample_time_s, i_vfd);	

		    }
		} else {
	            printf("ERROR: Invalid sample type: %d\n", sample_id);
	            return 1;
	        }
	    }
	
	   
	    fclose (fp);
	    free (stacks);
	    free (precise_functions);
    }

    double total_mpi_time = 0.0;
    print_stacktree (stack_tree->origin, 0, &total_mpi_time);
    free(hashes);

    return 0;
}

/**********************************************************************/


