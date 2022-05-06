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
//#include "vftr_scenarios.h"
#include "vftr_filewrite.h"

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
			double *t_avg, double *t_min, double *t_max,
			int *rank_min, int *rank_max, double *imbalance) {
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
				*rank_max = i;
			}
			if (all_times[i] < *t_min) {
				*t_min = all_times[i];
				*rank_min = i;
			}
		}
	}
	double d1 = *t_avg - *t_min;
	double d2 = *t_max - *t_avg;
	*imbalance = d1 > d2 ? d1 / *t_avg * 100 : d2 / *t_avg * 100;
}

/**********************************************************************/

// Auxiliary functions for colored output of MPI imbalance values

void set_red () {
	printf ("\033[0;31m");
}

void set_orange () {
	printf ("\033[0;33m");
}

void set_green () {
	printf ("\033[0;32m");
}

void reset_colors () {
	printf ("\033[0m");
}

/**********************************************************************/

void print_mpi_times (double t_avg, double t_min, double t_max,
		      int rank_min, int rank_max, double imbalance) {
	// MPI imbalances are highlighted in color
	printf (": MPI %4.3f %4.3f(%d) %4.3f(%d) ",
		t_avg, t_min, rank_min, t_max, rank_max);
	if (imbalance < 10) {
		set_green ();
	} else if (imbalance > 5 && imbalance < 50) {
		set_orange ();
	} else {
		set_red();
	}
	printf ("%4.2f %%\n", imbalance);
	reset_colors();
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
		int rank_min, rank_max;
		evaluate_mpi_time (leaf->time_spent,
				   &t_avg, &t_min, &t_max,
				   &rank_min, &rank_max, &imbalance);
		print_mpi_times (t_avg, t_min, t_max, rank_min, rank_max, imbalance);
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
	if (!leaf) return 0;
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

enum new_leaf_type {ORIGIN, NEXT, CALLEE};

void create_new_leaf (stack_leaf_t **new_leaf, char *name, enum new_leaf_type leaf_type) {
	if (leaf_type == ORIGIN) {
		*new_leaf = (stack_leaf_t*) malloc (sizeof(stack_leaf_t));
		(*new_leaf)->function_name = strdup(name);
		(*new_leaf)->module_name = "";
		(*new_leaf)->next_in_level = NULL;
		(*new_leaf)->callee = NULL;
		(*new_leaf)->origin = (stack_leaf_t*) malloc (sizeof(stack_leaf_t));
		(*new_leaf)->origin = *new_leaf;
		(*new_leaf)->entry_time = (double*)malloc (n_vfds * sizeof(double));
		(*new_leaf)->time_spent = (double*)malloc (n_vfds * sizeof(double));
		for (int i = 0; i < n_vfds; i++) {
			(*new_leaf)->entry_time[i] = 0.0;
			(*new_leaf)->time_spent[i] = 0.0;
		}
	} else if (leaf_type == NEXT) {
		(*new_leaf)->next_in_level = (stack_leaf_t*)malloc (sizeof(stack_leaf_t));
		(*new_leaf)->next_in_level->function_name = strdup(name);
		(*new_leaf)->next_in_level->module_name = "";
		(*new_leaf)->next_in_level->next_in_level = NULL;
		(*new_leaf)->next_in_level->callee = NULL;
		(*new_leaf)->next_in_level->origin = (stack_leaf_t*)malloc (sizeof(stack_leaf_t));
		(*new_leaf)->next_in_level->origin = (*new_leaf)->origin;
		(*new_leaf)->next_in_level->entry_time = (double*)malloc (n_vfds * sizeof(double));
		(*new_leaf)->next_in_level->time_spent = (double*)malloc (n_vfds * sizeof(double));
		for (int i = 0; i < n_vfds; i++) {
				(*new_leaf)->next_in_level->entry_time[i] = 0.0;
				(*new_leaf)->next_in_level->time_spent[i] = 0.0;
		}
	} else if (leaf_type == CALLEE) {
			(*new_leaf)->callee = (stack_leaf_t*) malloc (sizeof(stack_leaf_t));
			(*new_leaf)->callee->function_name = strdup(name);
			(*new_leaf)->callee->module_name = "";
			(*new_leaf)->callee->next_in_level = NULL;
			(*new_leaf)->callee->callee = NULL;
			(*new_leaf)->callee->origin = (stack_leaf_t*)malloc (sizeof(stack_leaf_t));
			(*new_leaf)->callee->origin = (*new_leaf)->origin;

			(*new_leaf)->callee->entry_time = (double*)malloc (n_vfds * sizeof(double));
			(*new_leaf)->callee->time_spent = (double*)malloc (n_vfds * sizeof(double));
			for (int i = 0; i < n_vfds; i++) {
					(*new_leaf)->callee->entry_time[i] = 0.0;
					(*new_leaf)->callee->time_spent[i] = 0.0;
			}
	}
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
		create_new_leaf (this_leaf, stacks[stackID].name, ORIGIN);
	}
	for (int level = n_stack_ids - 2; level >= 0; level--) {
		stackID = stack_ids[level];
		if ((*this_leaf)->callee) {
			*this_leaf = (*this_leaf)->callee;
			while (strcmp ((*this_leaf)->function_name, stacks[stackID].name)) {
				if ((*this_leaf)->next_in_level) {
					*this_leaf = (*this_leaf)->next_in_level;
				} else {
					create_new_leaf (this_leaf, stacks[stackID].name, NEXT);
					if (level == 0) {
						if (sample_id == SID_ENTRY) {
							(*this_leaf)->next_in_level->entry_time[i_vfd] = stime;
						} else {
							(*this_leaf)->next_in_level->time_spent[i_vfd] += (stime - (*this_leaf)->next_in_level->entry_time[i_vfd]);
						}
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
			create_new_leaf (this_leaf, stacks[stackID].name, CALLEE);
			if (level == 0) {
				if (sample_id == SID_ENTRY) {
					(*this_leaf)->callee->entry_time[i_vfd] = stime;
				} else {
					(*this_leaf)->callee->time_spent[i_vfd] += (stime - (*this_leaf)->callee->entry_time[i_vfd]);
				}
			}
			*this_leaf = (*this_leaf)->callee;
		}
	}
}

/**********************************************************************/

void show_progress (int i_vfd) {
	static int next_display = 0;
	if (next_display == 0) next_display = n_vfds / 4;
	if (i_vfd > next_display) {
		printf ("%d / %d\n", i_vfd, n_vfds);
		next_display += (n_vfds / 4);
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

    if (argc < 3) {
	    printf ("Usage: show_function_stacks <vfd-file> <search_func>\n");
	    return -1;
    }

    n_vfds = argc - 2;
    search_func = argv[n_vfds + 1];

    stack_leaf_t *stack_tree = NULL;

    printf ("Processing %d vfd files\n", n_vfds);

    for (int i_vfd = 0; i_vfd < n_vfds; i_vfd++) {

	    show_progress (i_vfd);
	    filename = argv[i_vfd+1];
	    fp = fopen (filename, "r");
	    assert (fp);

	    // We are not interested in the VFD version here
	    int dummy;
	    fread (&dummy, 1, sizeof(int), fp);
	    // From the header, we actually only need the stack and sample offset
	    read_fileheader (&vfd_header, fp);

	    // We need the number of hardware scenarios, because when scanning the samples
	    // and a message is encountered (sample_id == SID_MESSAGE), we need to scan over these
	    // values in order to be synchronized. Also, we allocate the corresponding (dummy-)buffer
	    fread (&(vfd_header.n_formulas), sizeof(int), 1, fp);
	    fread (&(vfd_header.n_hw_obs), sizeof(int), 1, fp);
            read_scenario_header (fp, vfd_header.n_hw_obs, vfd_header.n_formulas, false);

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

	    for (int i = 0; i < vfd_header.samplecount; i++ ) {
	        int sample_id;

	        fread (&sample_id, sizeof(int), 1, fp);

	        if (sample_id == SID_MESSAGE) {
		    skip_mpi_message_sample (fp);
	        } else if (sample_id == SID_ENTRY || sample_id == SID_EXIT) {
	            int stack_id;
		    long long sample_time, cycle_time;
		    read_stack_sample (fp, vfd_header.n_hw_obs, &stack_id, &sample_time, NULL, &cycle_time);
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

    return 0;
}

/**********************************************************************/


