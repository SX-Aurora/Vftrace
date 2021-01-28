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
    fread (&vfd_header->runtime, 1, sizeof(double), fp);
    fread (&vfd_header->function_samplecount, 1, sizeof(unsigned int), fp);
    fread (&vfd_header->message_samplecount, 1, sizeof(unsigned int), fp);
    fread (&vfd_header->stackscount, 1, sizeof(unsigned int), fp);
    fread (&vfd_header->stacksoffset, 1, sizeof(long), fp);
    fread (&vfd_header->sampleoffset, 1, sizeof(long), fp);
    vfd_header->samplecount = vfd_header->function_samplecount+vfd_header->message_samplecount;
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

void read_stacks (FILE *fp, stack_entry_t **stacks, function_entry_t **precise_functions, 
		  unsigned int stacks_count, long stacks_offset,
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
                if (!strcmp (record, (*precise_functions)[j].name)) {
                    (*stacks)[id].fun = j;
                    break;
                }
            }
	    
            if ((*stacks)[id].fun == -1 ) {
                (*n_precise_functions)++;
                if  (*n_precise_functions == 1) {
                    *precise_functions = (struct FunctionEntry *) malloc (*n_precise_functions * sizeof(struct FunctionEntry) );
                } else {
                    *precise_functions = (struct FunctionEntry *) realloc (*precise_functions, *n_precise_functions * sizeof(struct FunctionEntry) );
                }

                (*precise_functions)[*n_precise_functions - 1].name = strdup(record);
                (*precise_functions)[*n_precise_functions - 1].elapse_time = 0.0;
            }
        }

        for (int j = 0; j < *n_precise_functions; j++) {
            if (!strcmp ((*stacks)[id].name, (*precise_functions)[j].name)) {
                (*stacks)[id].fun = j;
                break;
            }
        }
    }
}

/**********************************************************************/

void print_fileheader (FILE *fp, vfd_header_t vfd_header) {
    int i;
    // We require a seperate datestring which is one element larger than
    // the field in the vfd file to add a terminating null character. This
    // is not necessary for the version string since it is written using sprintf
    // in the main code.
    char       datestring[25], record[RECORD_LENGTH], *s;
    datestring[24] = 0;
    strncpy (datestring, vfd_header.date, 24);

    fprintf (fp, "Version ID: %s\n", vfd_header.fileid    );
    fprintf (fp, "Date:       %s\n", datestring );
    fprintf (fp, "MPI tasks:  rank=%d count=%d\n", vfd_header.task, vfd_header.tasks); 
    fprintf (fp, "OpenMP threads:  thread=%d count=%d\n", vfd_header.thread, vfd_header.threads);
    fprintf (fp, "Sample interval: %12.6le seconds\n", vfd_header.interval*1.0e-6);
    fprintf (fp, "Job runtime:   %.3lf seconds\n", vfd_header.runtime);
    fprintf (fp, "Samples:       %d\n", vfd_header.samplecount );
    fprintf (fp, "   Function:   %d\n", vfd_header.function_samplecount );
    fprintf (fp, "   Messages:   %d\n", vfd_header.message_samplecount );
    fprintf (fp, "Unique stacks: %u\n", vfd_header.stackscount);
    fprintf (fp, "Stacks offset: %ld\n", vfd_header.stacksoffset);
    fprintf (fp, "Sample offset: %ld\n", vfd_header.sampleoffset);
}

/**********************************************************************/

// Reads the observable names and checks if it is integrated.
// Also, allocates the array which stores the hardware counter values.
void read_scenario_header (FILE *fp, int n_hw_obs, int n_formulas, bool verbose) {
	int slength;
        for (int i = 0; i < n_hw_obs; i++) {
           fread (&slength, sizeof(int), 1, fp);
           char *hw_obs_name = (char*)malloc(sizeof(char) * slength);
           fread (hw_obs_name, sizeof(char), slength, fp);
           fread (&slength, sizeof(int), 1, fp);
           char *variable_name = (char*)malloc(sizeof(char) * slength);
           fread (variable_name, sizeof(char), slength, fp);
           if (verbose) printf ("Hardware counter %d: %s, variable: %s\n", i, hw_obs_name, variable_name);
           free (hw_obs_name);
           free (variable_name);
        }
        for (int i = 0; i < n_formulas; i++) {
           fread (&slength, sizeof(int), 1, fp);
           char *formula_name = (char*) malloc (sizeof(char) * slength);
           fread (formula_name, sizeof(char), slength, fp);
           fread (&slength, sizeof(int), 1, fp);
           char *formula_expr = (char*) malloc (sizeof(char) * slength);
           fread (formula_expr, sizeof(char), slength, fp);
           bool is_integrated;
           fread (&is_integrated, sizeof(bool), 1, fp);
           if (verbose) printf ("%s: %s (%s)\n", formula_name, formula_expr, is_integrated ? "integrated" : "differential");
           free (formula_name);
           free (formula_expr);
        }
}

/**********************************************************************/

void read_mpi_message_sample (FILE *fp, int *direction, int *rank, int *type_index,
			      int *type_size, int *count, int *tag,
			      double *dt_start, double *dt_stop, double *rate,
                              int *callingStackID) {
	long long t_start, t_stop;
	fread (direction, sizeof(int), 1, fp);
	fread (rank, sizeof(int), 1, fp);
	fread (type_index, sizeof(int), 1, fp);
	fread (count, sizeof(int), 1, fp);
// The message size is given in bytes
	fread (type_size, sizeof(int), 1, fp);
	fread (tag, sizeof(int), 1, fp);
	fread (&t_start, sizeof(long long), 1, fp);
	fread (&t_stop, sizeof(long long), 1, fp);
// Convert microseconds to seconds
	*dt_start = t_start * 1.0e-6;
	*dt_stop = t_stop * 1.0e-6;
// Normalize to Megabytes.
	*rate = (*count) * (*type_size) / (*dt_stop - *dt_start) / (1024.0 * 1024.0);
        fread (callingStackID, sizeof(int), 1, fp);
}

/**********************************************************************/

// Skip the given number of integers and long longs. 
// If the MPI VFD entry format changes one day, just adapt these parameters.
void skip_mpi_message_sample (FILE *fp) {
#define N_MPI_SAMPLE_INT 7
#define N_MPI_SAMPLE_LONG 2
	struct {int dummy_i[N_MPI_SAMPLE_INT];
		long long dummy_l[N_MPI_SAMPLE_LONG];
	       }dummy;
	fread (&dummy, N_MPI_SAMPLE_INT * sizeof(int) + N_MPI_SAMPLE_LONG * sizeof(long long),
	       1, fp);
}

/**********************************************************************/

// Read the stack ID, which indicates from which path the function has been called.
// Read the time spent in this part of the program.
void read_stack_sample (FILE *fp, int n_hw_obs, int *stack_id,
			long long *sample_time, double *hw_values) {
	fread (stack_id, sizeof(int), 1, fp);
	fread (sample_time, sizeof(long long), 1, fp);
	for (int i = 0; i < n_hw_obs; i++) {
           fread (&(hw_values[i]), sizeof(double), 1, fp);
	}
}

/**********************************************************************/

// Skip the given number of integers and long longs.
// If the sample VFD entry format changes one day, just adapt these parameters.
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
