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
#include <string.h>

#include "vftr_scenarios.h"
#include "vftr_filewrite.h"
#include "vftr_vfd_utils.h"

void read_fileheader (vfd_header_t *vfd_header, FILE *fp) {
    fread (&vfd_header->fileid, 1, VFTR_FILEIDSIZE, fp);
    fread (&vfd_header->date, 1, 24, fp);
    fread (&vfd_header->interval, 1, sizeof(long long), fp);
    fread (&vfd_header->threads, 1, sizeof(int), fp);
    fread (&vfd_header->thread,	1, sizeof(int), fp);
    fread (&vfd_header->tasks, 1, sizeof(int), fp);
    fread (&vfd_header->task, 1, sizeof(int), fp);
    fread (&vfd_header->cycletime.l, 1, sizeof(long long), fp);
    fread (&vfd_header->inittime, 1, sizeof(long long), fp);
    fread (&vfd_header->runtime.l, 1, sizeof(long long), fp);
    fread (&vfd_header->samplecount, 1, sizeof(unsigned int), fp);
    fread (&vfd_header->stackscount, 1, sizeof(unsigned int), fp);
    fread (&vfd_header->stacksoffset, 1, sizeof(unsigned int), fp);
    fread (&vfd_header->sampleoffset, 1, sizeof(unsigned int), fp);
    fread (&vfd_header->reserved, 1, sizeof(unsigned int), fp);
}

/**********************************************************************/

bool is_precise (char *s) {
 	return s[strlen(s)-1] == '*';
}

/**********************************************************************/

char *strip_trailing_asterisk (char *s) {
	char *sdup = strdup(s);
	int n = strlen(sdup);
	if (sdup[n-1] == '*') {
		sdup[n-1] = '\0';
	}
	return sdup;
}

/**********************************************************************/

void read_stacks (FILE *fp, stack_entry_t **stacks, function_entry_t **functions, 
		  unsigned int stacks_count, unsigned int stacks_offset,
                  int *n_precise_functions, long *max_fp) {


    char record[RECORD_LENGTH];

    *stacks = (stack_entry_t *) malloc (stacks_count * sizeof(stack_entry_t));
    if (max_fp && ftell(fp) > *max_fp) {
       *max_fp = ftell(fp);
    }
    fseek (fp, stacks_offset, SEEK_SET);
    
    for (int i = 0; i < stacks_count; i++) {
	int id, levels, caller, len;
        fread (&id, sizeof(int), 1, fp);
	fread (&levels, sizeof(int), 1, fp);
	fread (&caller, sizeof(int), 1, fp);
	fread (&len, sizeof(int), 1, fp);
        len = len < RECORD_LENGTH ? len : RECORD_LENGTH - 1;
	fread (record, sizeof(char), len, fp);
	record[len] = '\0';
        (*stacks)[id].name = strdup(record);
	(*stacks)[id].levels = levels;
	(*stacks)[id].caller = caller;
	(*stacks)[id].precise = is_precise(record);

        (*stacks)[id].fun = -1;
        if ((*stacks)[id].precise) {
            for (int j = 0; j < *n_precise_functions; j++) {
                if (!strcmp (record, (*functions)[j].name)) {
                    (*stacks)[id].fun = j;
                    break;
                }
            }
	    
            if ((*stacks)[id].fun == -1 ) {
                (*n_precise_functions)++;
                if  (*n_precise_functions == 1) {
                    *functions = (struct FunctionEntry *) malloc (*n_precise_functions * sizeof(struct FunctionEntry) );
                } else {
                    *functions = (struct FunctionEntry *) realloc (*functions, *n_precise_functions * sizeof(struct FunctionEntry) );
                }

                (*functions)[*n_precise_functions - 1].name = strdup(record);
                (*functions)[*n_precise_functions - 1].elapse_time = 0.0;
            }
        }

        for (int j = 0; j < *n_precise_functions; j++) {
            if (!strcmp ((*stacks)[id].name, (*functions)[j].name)) {
                (*stacks)[id].fun = j;
                break;
            }
        }
    }
}

/**********************************************************************/

void print_fileheader (vfd_header_t vfd_header) {
    int i;
    // We require a seperate datestring which is one element larger than
    // the field in the vfd file to add a terminating null character. This
    // is not necessary for the version string since it is written using sprintf
    // in the main code.
    char       datestring[25], record[RECORD_LENGTH], *s;
    datestring[24] = 0;
    strncpy (datestring, vfd_header.date, 24);

    printf ("Version ID: %s\n", vfd_header.fileid    );
    printf ("Date:       %s\n", datestring );
    printf ("MPI tasks:  rank=%d count=%d\n", vfd_header.task, vfd_header.tasks); 
    printf ("OpenMP threads:  thread=%d count=%d\n", vfd_header.thread, vfd_header.threads);
    printf ("Sample interval: %12.6le seconds\n", vfd_header.interval*1.0e-6);
    printf ("Init time:     %lld\n", vfd_header.inittime);
    printf ("Job runtime:   %.3lf seconds\n", vfd_header.runtime.d);
    printf ("Samples:       %d\n", vfd_header.samplecount );
    printf ("Unique stacks: %d\n", vfd_header.stackscount);
    printf ("Stacks offset: %d\n", vfd_header.stacksoffset);
    printf ("Sample offset: %d\n", vfd_header.sampleoffset);
}

/**********************************************************************/

void init_hw_observables (FILE *fp, int n_hw_obs, double **hw_values) {
    	char name[SCENARIO_NAME_LEN];
	*hw_values = (double*)malloc (n_hw_obs * sizeof(double));
        for (int i = 0; i < n_hw_obs; i++) {
            	fread (name, SCENARIO_NAME_LEN, 1, fp);
            	printf ("Hardware observable name: %s\n", name);
            	int is_integrated;
            	fread (&is_integrated, sizeof(int), 1, fp);
            	printf ("Integrated counter: ");
            	is_integrated == 0 ? printf ("NO\n") : printf ("YES\n");
	        (*hw_values)[i] = 0.0;
        }
}

/**********************************************************************/

void skip_hw_observables (FILE *fp, int n_hw_obs) {
#define N_HW_SAMPLE_INT 1
	struct {char dummy_name[SCENARIO_NAME_LEN];
		int dummy_i[N_HW_SAMPLE_INT];
	       }dummy;
	for (int i = 0; i < n_hw_obs; i++) {
		fread (&dummy, SCENARIO_NAME_LEN + N_HW_SAMPLE_INT * sizeof(int), 1, fp);
	}
}

/**********************************************************************/

void read_mpi_message_sample (FILE *fp, int *direction, int *rank, int *type_index,
			      int *type_size, int *count, int *tag,
			      double *dt_start, double *dt_stop, double *rate) {
	long long t_start, t_stop;
	fread (direction, sizeof(int), 1, fp);
	fread (rank, sizeof(int), 1, fp);
	fread (type_index, sizeof(int), 1, fp);
	fread (count, sizeof(int), 1, fp);
	fread (type_size, sizeof(int), 1, fp);
	fread (tag, sizeof(int), 1, fp);
	fread (&t_start, sizeof(long long), 1, fp);
	fread (&t_stop, sizeof(long long), 1, fp);
	*dt_start = t_start * 1.0e-6;
	*dt_stop = t_stop * 1.0e-6;
	*rate = (*count) * (*type_size) / (*dt_stop - *dt_start) / (1024.0 * 1024.0);
}

/**********************************************************************/

void skip_mpi_message_sample (FILE *fp) {
#define N_MPI_SAMPLE_INT 6
#define N_MPI_SAMPLE_LONG 2
	struct {int dummy_i[N_MPI_SAMPLE_INT];
		long long dummy_l[N_MPI_SAMPLE_LONG];
	       }dummy;
	fread (&dummy, N_MPI_SAMPLE_INT * sizeof(int) + N_MPI_SAMPLE_LONG * sizeof(long long),
	       1, fp);
}

/**********************************************************************/

void read_stack_sample (FILE *fp, int n_hw_obs, int *stack_id,
			long long *sample_time, double **hw_values) {
	fread (stack_id, sizeof(int), 1, fp);
	fread (sample_time, sizeof(long long), 1, fp);
	for (int i = 0; i < n_hw_obs; i++) {
		fread (hw_values[i], sizeof(double), 1, fp);
	}
}

/**********************************************************************/

void skip_stack_sample (FILE *fp) {
#define N_STACK_SAMPLE_INT 1
#define N_STACK_SAMPLE_LONG 1
	struct {int dummy_i[N_STACK_SAMPLE_INT];
		long long dummy_l[N_STACK_SAMPLE_LONG];
	       }dummy;
	fread (&dummy, N_STACK_SAMPLE_INT * sizeof(int) + N_STACK_SAMPLE_LONG * sizeof(long long),
	       1, fp);	
}

/**********************************************************************/
