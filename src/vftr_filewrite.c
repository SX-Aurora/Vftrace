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

#define _GNU_SOURCE

#include <string.h>
#include <time.h>
#include <assert.h>
#include <signal.h>
#include <math.h>
#include <limits.h>

#include "vftr_scenarios.h"
#include "vftr_hwcounters.h"
#include "vftr_environment.h"
#include "vftr_hooks.h"
#include "vftr_fileutils.h"
#include "vftr_filewrite.h"
#include "vftr_functions.h"
#include "vftr_timer.h"
#include "vftr_setup.h"
#include "vftr_mpi_utils.h"
#include "vftr_stacks.h"

// File pointer of the log file
FILE *vftr_log = NULL;

// Individual vftrace-internal file id
char vftr_fileid[VFTR_FILEIDSIZE];

// The next time step where a snapshot is written to the vfd file
long long vftr_nextsampletime;

// We only need this variable to create the .log and .vfd file names.
// It is global because it must be created before MPI_Init is called.
// This is because the program path is determined by opening the file
// /proc/<pid>/cmdline, where a race condition can occur if multiple
// ranks access it.
char *vftr_program_path;

// The basename of Vftrace log files
char *vftr_logfile_name;

FILE *vftr_vfd_file;

// TODO: Explain
long vftr_admin_offset;
long vftr_samples_offset;

/**********************************************************************/

char *vftr_get_program_path () {
	char *basename;
	// User-defined output file
	if (vftr_environment.logfile_basename->set) {
		basename = vftr_environment.logfile_basename->value;
	} else {
		// program_path is either <abs_path>/app_name or ./app_name
		char *program_path = vftr_get_application_name ();
		char *s;
		if (program_path) {
		  // rindex returns a pointer to the last occurence of '/'
		  if (s = rindex (program_path, '/')) {
		  	// The basename is everything after the last '/'
		  	basename = strdup (s + 1);
		  } else {
		  	basename = strdup (program_path);
		  }
		} else {
		  basename = strdup ("unknown");
		}
	}
	return basename;
}

/**********************************************************************/

// Creates the outputfile name of the form <vftr_program_path>_<mpi_rank>.out.
// <vftr_program_path> is either the application name or a value defined by 
// the user in the environment variable LOGFILE_BASENAME. It is a global variable
// obtained beforehand by a call to vftr_get_program_path to avoid a race condition.
// Suffix is ".log" for ASCII log files and ".vfd" for viewer files.
char *vftr_create_logfile_name (int mpi_rank, int mpi_size, char *suffix) {
	bool read_from_env = false;
	// The user can also define a different output directory
	char *out_directory;
	if (vftr_environment.output_directory->set) {
		out_directory = vftr_environment.output_directory->value;
	} else {
		out_directory = strdup (".");
	}
	
	vftr_program_path = vftr_get_program_path ();	
	// Finally create the output file name
	int task_digits = vftr_count_digits (mpi_size);
	char *logfile_nameformat = (char*)malloc (1024 * sizeof(char));
	sprintf (logfile_nameformat, "%s/%s_%%0%dd.%s",
		 out_directory, vftr_program_path, task_digits, suffix);
	char *logfile_name = (char*)malloc (1024 * sizeof(char));
	sprintf (logfile_name, logfile_nameformat, mpi_rank);
	free (logfile_nameformat);
	return logfile_name;
}

/**********************************************************************/

void vftr_init_vfd_file () {
	const int one = 1;
	char *filename = vftr_create_logfile_name (vftr_mpirank, vftr_mpisize, "vfd");
	FILE *fp = fopen (filename, "w+");
	assert (fp);
	size_t size = vftr_environment.bufsize->value * 1024 * 1024;
	char *buf = (char *) malloc (size);
	assert (buf);
	int status = setvbuf (fp, buf, _IOFBF, size);
	assert (!status);
	
	/* Some of these numbers will be updated later in vftr_finalize */
	
	int vfd_version = VFD_VERSION;
    	time_t current_date;
	time (&current_date);
	char *datestring = ctime (&current_date);

	fwrite (&vfd_version, sizeof(int), 1, fp);
	fwrite (vftr_fileid, VFTR_FILEIDSIZE, 1, fp );   
	fwrite (datestring, 24, 1, fp );    
	fwrite (&vftr_interval, sizeof(long long), 1, fp );
	fwrite (&one, sizeof(int), 1, fp ); 
	int omp_thread = 0;
	fwrite (&omp_thread, sizeof(int), 1, fp ); 
	
	if (omp_thread == 0) vftr_admin_offset = ftell (fp);
	
        // Reserve space for the following observables. Their explicit values are determined later.
    	unsigned int zeroint[] = {0, 0, 0, 0, 0, 0, 0};
    	double zerodouble[] = {0., 0.};
    	long long zerolong[] = {0};
	// vftr_mpi_size and vftr_mpi_rank
	fwrite (zeroint, sizeof(unsigned int), 2, fp);
	// vftr_cycletime
	fwrite (zerodouble, sizeof(double), 1, fp);
	// vftr_inittime
	fwrite (zerolong, sizeof(long long), 1, fp);
  	// vftr runtime
	fwrite (zerodouble, sizeof(double), 1, fp);
        // Five integers: sample_count, stacks_count, stacks_offset, sample_offset and ???
	fwrite (zeroint, sizeof(unsigned int), 2, fp);
	fwrite (zerolong, sizeof(long), 1, fp);
	fwrite (zerolong, sizeof(long), 1, fp);
	fwrite (zerolong, sizeof(long), 1, fp);
	// Store global information about hardware scenarios
	vftr_write_scenario_header_to_vfd (fp);	

	if (omp_thread == 0) vftr_samples_offset = ftell (fp);
	vftr_vfd_file = fp;
}

/**********************************************************************/

void vftr_finalize_vfd_file (long long finalize_time, int signal_number) {
    if (vftr_env_do_sampling () && signal_number != SIGUSR1) {

        long stackstable_offset = ftell (vftr_vfd_file);
        vftr_write_stacks_vfd (vftr_vfd_file, 0, vftr_froots);

        // It is unused ?
        long profile_offset = 0;

        double runtime = finalize_time * 1.0e-6;
        double zerodouble[] = { 0., 0. };
    	long long zerolong[] = {0};

        // Update trace info in header and close
        fseek (vftr_vfd_file, vftr_admin_offset, SEEK_SET);
        fwrite (&vftr_mpisize, sizeof(int), 1, vftr_vfd_file); 
        fwrite (&vftr_mpirank, sizeof(int),1, vftr_vfd_file); 
        fwrite (zerodouble, sizeof(double),	1, vftr_vfd_file); 
	// vftr_inittime has to be removed from the vfd file format, as
	// it is always zero! But we have to check it against the viewer.
        // fwrite(&vftr_inittime, sizeof(long long), 1, vftr_vfd_file);
	fwrite (zerolong, sizeof(long long), 1, vftr_vfd_file);
        fwrite (&runtime, sizeof(double), 1, vftr_vfd_file);
        fwrite (&vftr_samplecount, sizeof(unsigned int), 1, vftr_vfd_file);
        fwrite (&vftr_stackscount, sizeof(unsigned int), 1, vftr_vfd_file);
        fwrite (&stackstable_offset, sizeof(long), 1, vftr_vfd_file);
        fwrite (&vftr_samples_offset, sizeof(long), 1, vftr_vfd_file);
        fwrite (&profile_offset, sizeof(long), 1, vftr_vfd_file);
        fclose (vftr_vfd_file);
    }
}

/**********************************************************************/

void vftr_write_to_vfd(long long runtime, unsigned long long cycles, int stack_id, unsigned int sid) {
    fwrite (&sid, sizeof(unsigned int), 1, vftr_vfd_file);
    fwrite (&stack_id, sizeof(int), 1, vftr_vfd_file);
    fwrite (&runtime, sizeof(long long), 1, vftr_vfd_file);

    vftr_write_observables_to_vfd (cycles, vftr_vfd_file);

    vftr_nextsampletime = runtime + vftr_interval;
    vftr_prevsampletime = runtime;
    vftr_samplecount++;
}

/**********************************************************************/

#ifdef _MPI
// store some message information for use in the log file
void vftr_log_message_info(vftr_direction dir, int count, int type_idx,
                           int type_size, int rank, int tag,
                           long long tstart, long long tend) {
   // vftr_stack is the current function stack pointer
   vftr_fstack;

   if (dir == send) {
      vftr_fstack->prof_current.mpi_tot_send_bytes += count * type_size;
   } else {
      vftr_fstack->prof_current.mpi_tot_recv_bytes += count * type_size;
   }
}

// Store the message information in a vfd file
void vftr_store_message_info(vftr_direction dir, int count, int type_idx,
                             int type_size, int rank, int tag,
                             long long tstart, long long tend) {
   
   int sid = SID_MESSAGE;
   fwrite(&sid, sizeof(int), 1, vftr_vfd_file);
   fwrite(&dir, sizeof(int), 1, vftr_vfd_file);
   fwrite(&rank, sizeof(int), 1, vftr_vfd_file);
   fwrite(&type_idx, sizeof(int), 1, vftr_vfd_file);
   fwrite(&count, sizeof(int), 1, vftr_vfd_file);
   fwrite(&type_size, sizeof(int), 1, vftr_vfd_file);
   fwrite(&tag, sizeof(int), 1, vftr_vfd_file);
   fwrite(&tstart, sizeof(long long), 1, vftr_vfd_file);
   fwrite(&tend, sizeof(long long), 1, vftr_vfd_file);

   vftr_samplecount++;
}
#endif

/**********************************************************************/

void vftr_write_profile () {
    const int zero = 0;
    double         rtime;
    unsigned long long      total_cycles, calls, cycles, *ec;
    evtcounter_t    *evc;
    FILE           *fp = vftr_vfd_file;

    function_t   **funcTable;

    if (!vftr_stackscount)return;
    funcTable = vftr_func_table;

    ec = (unsigned long long *) malloc (vftr_n_hw_obs * sizeof(long long));
    for (int j = 0; j < vftr_n_hw_obs; j++) {
	ec[j] = 0;
    }

    total_cycles = 0;
 
    /* Sum all cycles and counts */
    for (int i = 0; i < vftr_stackscount; i++) {
	if (funcTable[i] && funcTable[i]->return_to && funcTable[i]->prof_current.calls) {
            profdata_t *prof_current = &funcTable[i]->prof_current;
	    total_cycles += prof_current->cycles;
            if (!prof_current->event_count) continue;
            for (int j = 0; j < vftr_n_hw_obs; j++) {
                ec[j] += prof_current->event_count[j];
	    }
	}
    }
    rtime  = vftr_get_runtime_usec() * 1.0e-6;

    /* Write profile info */
    
    fwrite (&rtime, sizeof(double), 1, fp);  /* Application time */
    fwrite (&zero, sizeof(int), 1, fp);  /* Scenario */
    fwrite (&vftr_n_hw_obs, sizeof(int), 1, fp);  /* Nr of events */
    for (evc = vftr_get_counters();  evc; evc=evc->next ) {
        fwrite( &(evc->namelen), sizeof(int),       1, fp );  /* Event namelength */
        fwrite( &(evc->name),    sizeof(int),       8, fp );  /* Event name */
    }
    fwrite(  ec,                 sizeof(long long), vftr_n_hw_obs, 
                                                       fp );  /* Raw total event counts */

    for (int i = 0; i < vftr_stackscount; i++) {
        if (!funcTable[i]) continue;
        profdata_t *prof_current  = &funcTable[i]->prof_current;
	calls  = prof_current->calls ;
	cycles = prof_current->calls ? prof_current->cycles : 0;
        fwrite (&calls, sizeof(long long), 1, fp);  /* Calls */
        fwrite (&cycles, sizeof(long long), 1, fp);  /* Cycles */
        if (prof_current->event_count) {
            fwrite (&vftr_n_hw_obs, sizeof(int), 1, fp);  /* Nr of events */
            fwrite (prof_current->event_count,
	    	sizeof(long long), vftr_n_hw_obs, fp);  /* Event counts */
        } else {
            fwrite (&zero, sizeof(int), 1, fp );  /* No events: write 0 */
        }
    }
}

/**********************************************************************/

// Outputs the header of a single column. Multiple calls of this function generate the
// entire header. The entire string length is "largest_column_length". We print "header",
// and fill up all characters up to "largest_column_length" with empty spaces. This way,
// all elements of this column start at the same position. 
void output_header (char *header, int largest_column_length, FILE *fp) {
	char *sh = header;
	int name_length = strlen(header);
	int n_chars = name_length < largest_column_length ? name_length : largest_column_length;
	int n_fillup = name_length < largest_column_length ? largest_column_length - name_length + 1  : 1;
	for (int i = 0; i < n_chars; i++) {
		fputc (*sh++, fp);
	}
	for (int i = 0; i < n_fillup; i++) {
		fputc (' ', fp);
	}
}

/**********************************************************************/

// Compute the column width by checking how many times value can be divided by 10. If this number
// exceeds the existing width value, overwrite it.
void compute_column_width (long long value, int *width) {
	int count, this_width;
	for (count = value, this_width = 0; count; count /= 10, this_width++);
	if (this_width > *width) *width = this_width;
}

/**********************************************************************/

// Prints n_dashes "-"
void output_dashes_nextline (int n_dashes, FILE *fp) {
	for (int j = 0; j < n_dashes; j++) {
		fputc ('-', fp);
	}
	fputc ('\n', fp);
}

/**********************************************************************/

void print_stack_time (FILE *fp, int calls, char *fmttime, char *fmttimeInc, float t_excl, float t_incl, float t_part, float t_cumm) {

	float stime = calls ? t_excl : 0;
        fprintf (fp, fmttime, stime);
	stime  = calls ? t_incl : 0;
        fprintf (fp, fmttimeInc , stime < 0. ? 0. : stime);
        fprintf (fp, t_part < 99.95 ? "%4.1f " : "100. ", t_part);
        fprintf (fp, t_cumm < 99.95 ? "%4.1f " : "100. ", t_cumm);
}

/**********************************************************************/

void set_evc_decipl (int n_indices, int n_scenarios, evtcounter_t *evc1, evtcounter_t *evc) {
    int e;
    for (int i = 0; i < n_indices; i++) {
        for (e = 0, evc = evc1; evc; e++, evc = evc->next) {
            compute_column_width (scenario_expr_counter_values[e], &evc->decipl);
        }
    }
}

/**********************************************************************/

void get_stack_times (profdata_t *prof_current, profdata_t *prof_previous, float runtime,
                      float *t_excl, float *t_incl, float *t_part) {
	long long timeExcl_usec = prof_current->timeExcl - prof_previous->timeExcl;
	long long timeIncl_usec = prof_current->timeIncl - prof_previous->timeIncl;
	*t_excl = timeExcl_usec * 1.0e-6;
	*t_incl = timeIncl_usec * 1.0e-6;
	*t_part = *t_excl * 100.0 / runtime;
}

/**********************************************************************/

void fill_indices_to_evaluate (function_t **funcTable, double runtime, int *indices) {
	float ctime = 0.;
    	float max_ctime = 99.;
	float t_excl, t_incl, t_part;
	int j = 0;
	ctime = 0.;
	for (int i = 0; i < vftr_stackscount; i++) {
		if (funcTable[i] == NULL) continue;
		profdata_t *prof_current = &funcTable[i]->prof_current;
		profdata_t *prof_previous = &funcTable[i]->prof_previous;
		/* If function has a caller and has been called */
		if (!(funcTable[i]->return_to && prof_current->calls)) continue;
		indices[j++] = i;
		get_stack_times (prof_current, prof_previous, runtime, &t_excl, &t_incl, &t_part);
		ctime += t_part;
		if (vftr_environment.prof_truncate->value && ctime > max_ctime) break;
	}
}

/**********************************************************************/

int count_indices_to_evaluate (function_t **funcTable, double runtime) {
	int n_indices = 0;
	float ctime = 0.;
    	float max_ctime = 99.;
	float t_excl, t_incl, t_part;
	for (int i = 0; i < vftr_stackscount; i++) {
		if (funcTable[i] == NULL) continue;
		profdata_t *prof_current = &funcTable[i]->prof_current;
		profdata_t *prof_previous = &funcTable[i]->prof_previous;
		/* If function has a caller and has been called */
		if (!(funcTable[i]->return_to && prof_current->calls)) continue;
		
		n_indices++;

		get_stack_times (prof_current, prof_previous, runtime, &t_excl, &t_incl, &t_part);
		ctime += t_part;
		if (vftr_environment.prof_truncate->value && ctime > max_ctime) break;
	}
	return n_indices;
}

/**********************************************************************/

void fill_scenario_counter_values (double *val, int n_vars, profdata_t *prof_current, profdata_t *prof_previous) {
	memset (scenario_expr_counter_values, 0., sizeof (double) * scenario_expr_n_vars);
	if (prof_current->event_count) {
		for (int i = 0; i < n_vars; i++) {
			val[i] += (double)prof_current->event_count[i];
		}
	}
        if (prof_previous->event_count) {
		for (int i = 0; i < n_vars; i++) {
			val[i] -= (double)prof_previous->event_count[i];
		}
	}
}

/**********************************************************************/

#define MIN_CALLS_NCHAR 3
#define MIN_FUNC_NCHAR 1
#define MIN_CALLER_NCHAR 5
#define MIN_EXCLTIME_NCHAR 1
#define MIN_INCTIME_NCHAR 1

void set_formats (function_t **funcTable, double runtime,
		   int n_indices, int *indices, format_t *format) {
	long long ev;
	for (format->fid = 0, ev = vftr_gStackscount; ev; ev /= 10, format->fid++);
	for (format->rank = 0, ev = vftr_mpisize; ev; ev /= 10, format->rank++);
	format->fid = 2;
	format->rank = 2;
	format->n_calls = MIN_CALLS_NCHAR;
	format->func_name = MIN_FUNC_NCHAR;
	format->caller_name = MIN_CALLER_NCHAR;
	format->incl_time = MIN_INCTIME_NCHAR;
        format->excl_time = MIN_EXCLTIME_NCHAR;
	// 
	for (int i = 0; i < n_indices; i++) {
		int i_func = indices[i];
		profdata_t *prof_current = &funcTable[i_func]->prof_current;
		profdata_t *prof_previous = &funcTable[i_func]->prof_previous;

		if (vftr_events_enabled) {
			fill_scenario_counter_values (scenario_expr_counter_values,
				scenario_expr_n_vars, prof_current, prof_previous);
		}

        	int k = strlen(funcTable[i_func]->name);
		if (k > format->func_name) format->func_name = k;
		function_t *func;
        	if (func = funcTable[i_func]->return_to) {
        	    k = strlen(func->name);
		    if (k > format->caller_name) format->caller_name = k;
        	}

        	int calls  = prof_current->calls - prof_previous->calls;
		
		float t_excl, t_incl, t_part;
		get_stack_times (prof_current, prof_previous, runtime, &t_excl, &t_incl, &t_part);

        	compute_column_width (calls, &(format->n_calls));
        	compute_column_width (t_excl * 10000., &(format->excl_time));
        	compute_column_width (t_incl * 10000., &(format->incl_time));

		if (vftr_events_enabled) {
		    unsigned long long cycles = prof_current->cycles - prof_previous->cycles;
		    scenario_expr_evaluate_all (t_excl, cycles);
		    scenario_expr_set_formats ();
	        }
	}
	if (format->excl_time < 5) format->excl_time = 5;
    	if (format->incl_time < 5) format->incl_time = 5;
}

/**********************************************************************/

#ifdef _MPI
double compute_mpi_imbalance (long long *all_times, double t_avg) {
	double max_diff = 0;
	// If no average time is given (e.g. when it is not of interest), compute it here
	if (t_avg < 0.0) {
		long long sum_times = 0ll;
		int n = 0;
		for (int i = 0; i < vftr_mpisize; i++) {
			if (all_times[i] > 0) {
				n++;
				sum_times += all_times[i];
			}
		}
		if (n > 0) {
			t_avg = (double)sum_times / n;
		} else {
			return 0;
		}
	}
	for (int i = 0; i < vftr_mpisize; i++) {
		if (all_times[i] > 0) {
			double d = fabs((double)(all_times[i]) - t_avg);
			if (d > max_diff) max_diff = d;
		}
	}
	return max_diff / t_avg * 100;
}

/**********************************************************************/

void evaluate_display_function (char *func_name, display_function_t **display_func,
				bool display_sync_time) {
    char func_name_sync[strlen(func_name)+5];
    int n_func_indices, n_stack_indices;
    int *stack_indices = NULL, *func_indices = NULL;	
    int n_func_indices_sync, n_stack_indices_sync;
    int *func_indices_sync = NULL, *stack_indices_sync = NULL;

    vftr_find_function_in_table (func_name, &func_indices, &n_func_indices, true);
    (*display_func)->n_func_indices = n_func_indices;
    (*display_func)->func_indices = (int*)malloc (n_func_indices * sizeof(int));
    memcpy ((*display_func)->func_indices, func_indices, n_func_indices * sizeof(int));


    vftr_find_function_in_stack (func_name, &stack_indices, &n_stack_indices, true);
    (*display_func)->n_stack_indices = n_stack_indices;
    (*display_func)->stack_indices = (int*)malloc (n_stack_indices * sizeof(int));
    memcpy ((*display_func)->stack_indices, stack_indices, n_stack_indices * sizeof(int));

    if (display_sync_time) {
    	strcpy (func_name_sync, func_name);
    	strcat (func_name_sync, "_sync");
	vftr_find_function_in_table (func_name_sync, &func_indices_sync, &n_func_indices_sync, true);
    	if (n_func_indices_sync > 0 && n_func_indices != n_func_indices_sync) {
    	    printf ("Error: Number of synchronize regions does not match total number of regions: %d %d\n",
    	    	n_func_indices, n_func_indices_sync);
    	}
	vftr_find_function_in_stack (func_name_sync, &stack_indices_sync, &n_stack_indices_sync, true);
    	if (n_stack_indices_sync > 0 && n_stack_indices != n_stack_indices_sync) {
    	    printf ("Error: Number of synchronize regions does not match total number of regions: %d %d\n",
    	    	n_stack_indices, n_stack_indices_sync);
    	}

    } else {
	n_func_indices_sync = 0;
	n_stack_indices_sync = 0;
    }

    (*display_func)->this_mpi_time = 0;
    (*display_func)->this_sync_time = 0;
    (*display_func)->n_calls = 0;
    (*display_func)->mpi_tot_send_bytes = 0;
    (*display_func)->mpi_tot_recv_bytes = 0;
    for (int i = 0; i < n_func_indices; i++) {
	(*display_func)->this_mpi_time += vftr_func_table[func_indices[i]]->prof_current.timeIncl;
	if (n_func_indices_sync > 0) (*display_func)->this_sync_time += vftr_func_table[func_indices_sync[i]]->prof_current.timeIncl;
	(*display_func)->n_calls += vftr_func_table[func_indices[i]]->prof_current.calls;
	(*display_func)->mpi_tot_send_bytes += vftr_func_table[func_indices[i]]->prof_current.mpi_tot_send_bytes;
	(*display_func)->mpi_tot_recv_bytes += vftr_func_table[func_indices[i]]->prof_current.mpi_tot_recv_bytes;
    }
    long long all_times [vftr_mpisize], all_times_sync [vftr_mpisize];
    PMPI_Allgather (&(*display_func)->this_mpi_time, 1, MPI_LONG_LONG_INT, all_times,
		 1, MPI_LONG_LONG_INT, MPI_COMM_WORLD);
    if (display_sync_time) {
	    PMPI_Allgather (&(*display_func)->this_sync_time, 1, MPI_LONG_LONG_INT, all_times_sync,
			 1, MPI_LONG_LONG_INT, MPI_COMM_WORLD);
    }
    (*display_func)->t_max = 0;
    (*display_func)->t_sync_max = 0;
    (*display_func)->t_min = LLONG_MAX;
    (*display_func)->t_sync_min = LLONG_MAX;
    (*display_func)->t_avg = 0.0;
    (*display_func)->t_sync_avg = 0.0;
    (*display_func)->imbalance = 0.0;

    long long sum_times = 0;
    long long sum_times_sync = 0;
    int n_count = 0;
    if ((*display_func)->n_calls == 0) return;
    for (int i = 0; i < vftr_mpisize; i++) {
    	if (all_times[i] > 0) {
    		sum_times += all_times[i];
		if (n_func_indices_sync > 0) sum_times_sync += all_times_sync[i];
    		n_count++;
    	}
    }
    if (n_count > 0) {
       (*display_func)->t_avg = (double)sum_times / n_count;
       if (n_func_indices_sync > 0) (*display_func)->t_sync_avg = (double)sum_times_sync / n_count;
       (*display_func)->imbalance = compute_mpi_imbalance (all_times, (*display_func)->t_avg);
       for (int i = 0; i < vftr_mpisize; i++) {	
       	  if (all_times[i] > 0) {
       		if (all_times[i] < (*display_func)->t_min) {
			(*display_func)->t_min = all_times[i];
			if (n_func_indices_sync > 0) (*display_func)->t_sync_min = all_times_sync[i];
		}
       		if (all_times[i] > (*display_func)->t_max) {
			(*display_func)->t_max = all_times[i];
			if (n_func_indices_sync > 0) (*display_func)->t_sync_max = all_times_sync[i];
		}
       	  }
       }
    }
}

/**********************************************************************/

int vftr_compare_display_functions_tavg (const void *a1, const void *a2) {
	display_function_t *f1 = *(display_function_t **)a1;
	display_function_t *f2 = *(display_function_t **)a2;
	if (!f2) return -1;
	if (!f1) return 1;
	double t1 = f1->t_avg;
	double t2 = f2->t_avg;
	double diff = t2 - t1;
	if (diff > 0) return 1;
	if (diff < 0) return -1;
	return 0; 
}

int vftr_compare_display_functions_iorig (const void *a1, const void *a2) {
	display_function_t *f1 = *(display_function_t **)a1;
	display_function_t *f2 = *(display_function_t **)a2;
	if (!f2) return -1;
	if (!f1) return 1;
	double t1 = f1->i_orig;
	double t2 = f2->i_orig;
	double diff = t1 - t2; // Small indices first
	if (diff > 0) return 1;
	if (diff < 0) return -1;
	return 0;
}

/**********************************************************************/

void vftr_get_display_width (display_function_t **display_functions,
				int n_display_functions, int n_decimal_places,
				int *n_func_max, int *n_calls_max,
				int *n_t_avg_max, int *n_t_min_max,
				int *n_t_max_max, int *n_imba_max, int *n_t_max) {

	// Loop over all display functions with calls and determine the maximum
	// value of digits or characters required to represent the corresponding
	// field value.
	// For floating point numbers, only the digits in front of the comma are counted.
	// and n_decimal_places + 1 (for the comma itself) is added.
        *n_func_max = 0;
        *n_calls_max = 0;
        *n_t_avg_max = 0;
        *n_t_min_max = 0;
        *n_t_max_max = 0;
        *n_imba_max = 0;
        *n_t_max = 0;


	int n;
	for (int i = 0; i < n_display_functions; i++) {
		if (display_functions[i]->n_calls > 0) {
			n = strlen (display_functions[i]->func_name);
			if (n > *n_func_max) *n_func_max = n;
			n = vftr_count_digits (display_functions[i]->n_calls);
			if (n > *n_calls_max) *n_calls_max = n;	
			n = vftr_count_digits_double (display_functions[i]->t_avg * 1e-6);
			if (n > *n_t_avg_max) *n_t_avg_max = n;
			n = vftr_count_digits_double (display_functions[i]->t_min * 1e-6);
			if (n > *n_t_min_max) *n_t_min_max = n;
			n = vftr_count_digits_double (display_functions[i]->t_max * 1e-6);
			if (n > *n_t_max_max) *n_t_max_max = n;
			n = vftr_count_digits_double (display_functions[i]->imbalance);
			if (n > *n_imba_max) *n_imba_max = n;
			n = vftr_count_digits_double (display_functions[i]->this_mpi_time * 1e-6);
			if (n > *n_t_max) *n_t_max = n;
		}
	}
	*n_t_avg_max += n_decimal_places + 1;
	*n_t_min_max += n_decimal_places + 1;
	*n_t_max_max += n_decimal_places + 1;
	*n_imba_max += n_decimal_places + 1;
	*n_t_max += n_decimal_places + 1;
}

/**********************************************************************/

void vftr_print_function_statistics (FILE *pout, bool display_sync_time, 
				     char *display_function_names[], int n_display_functions) {

    display_function_t **display_functions =
			(display_function_t**) malloc (n_display_functions * sizeof(display_function_t*));

    for (int i = 0; i < n_display_functions; i++) {
	display_functions[i] = (display_function_t*) malloc (sizeof(display_function_t));
	display_functions[i]->func_name = strdup(display_function_names[i]);
        display_functions[i]->i_orig = i;
    }
    
    double total_time = 0;
    for (int i = 0; i < n_display_functions; i++) {
       evaluate_display_function (display_function_names[i], &(display_functions[i]), display_sync_time);
       total_time += display_functions[i]->this_mpi_time * 1e-6;
    }

    qsort ((void*)display_functions, (size_t)n_display_functions,
	    sizeof (display_function_t *), vftr_compare_display_functions_tavg);


    fprintf (pout, "Total time spent in MPI for rank %d: %lf s\n", vftr_mpirank, total_time);
    fprintf (pout, "Imbalance computed as: max (T - T_avg)\n");

    // Most of this code deals with the determination of the column widths.
    // vftr_get_display_width loops over all display_functions and determines
    // the maximum number of digits or characters required to display the
    // corresponding column. 
    int n_func, n_calls, n_t_avg, n_t_min, n_t_max, n_imba, n_t;
    vftr_get_display_width (display_functions, n_display_functions, 5,
	&n_func, &n_calls, &n_t_avg, &n_t_min, &n_t_max, &n_imba, &n_t);

    // The following headers appear in the function table but also define
    // the minimum widths for each column. Below, we check if the previously
    // computed widths are below these default values and increase them
    // if necessary. 
    // One complication arises through the display of the percentage of the time
    // spent in the synthetic synchronization barrier for collective MPI calls.
    // It is printed directly behind the associated absolute time in brackets "(%...)",
    // requiring 8 characters. This value is added to the column width when 
    // comparing to the default column widths.
    const char *headers[10] = {"Function", "%MPI", "Calls",
                              "Total send ", "Total recv.",
			      "Avg. time [s]", "Min. time [s]", "Max. time [s]",
			      "Imbalance", "This rank [s]"};
    enum column_ids {FUNC, MPI, CALLS, TOT_SEND_BYTES, TOT_RECV_BYTES, T_AVG, T_MIN, T_MAX, IMBA, THIS_T};

    // Note that there is no treatment of the width of the %MPI column, since the value
    // inside can never exceed 99.99%. Therefore, it has a fixed length of 5.
    int n_func_0 = strlen(headers[FUNC]);
    int n_calls_0 = strlen(headers[CALLS]);
    int n_tot_send_bytes = strlen(headers[TOT_SEND_BYTES]);
    int n_tot_recv_bytes = strlen(headers[TOT_RECV_BYTES]);
    int n_t_avg_0 = strlen (headers[T_AVG]);
    int n_t_min_0 = strlen (headers[T_MIN]);
    int n_t_max_0 = strlen (headers[T_MAX]);
    int n_imba_0 = strlen (headers[IMBA]);
    int n_t_0 = strlen(headers[THIS_T]);

    int add_sync_spaces = display_sync_time ? 8 : 0;

    if (n_func < n_func_0) n_func = n_func_0;
    if (n_calls < n_calls_0) n_calls = n_calls_0;
    if (n_t_avg + add_sync_spaces < n_t_avg_0) {
	n_t_avg = n_t_avg_0;
    } else {
	n_t_avg += add_sync_spaces;
    }
    if (n_t_min + add_sync_spaces < n_t_min_0) {
	n_t_min = n_t_min_0;
    } else {
	n_t_min += add_sync_spaces;
    }
    if (n_t_max + add_sync_spaces < n_t_max_0) {
	n_t_max = n_t_max_0;
    } else {
	n_t_max += add_sync_spaces;
    }
    if (n_imba < n_imba_0) n_imba = n_imba_0;
    if (n_t + add_sync_spaces < n_t_0) {
	n_t = n_t_0;
    } else {
	n_t += add_sync_spaces;
    }

    // We compute the total width of the MPI table to print separator lines.
    // There are the widths computed above, as well as the width of the MPI field,
    // which is 6. Additionally, there are 11 "|" characters and 20 spaces around them.
    // So in total, we have a fixed summand of 11 + 20 = 31.
    int n_spaces_tot = n_func + 6 + n_calls + n_tot_send_bytes + n_tot_recv_bytes + n_t_avg + n_t_min + n_t_max + n_imba + n_t + 31;

    // Print a separator line ("----------"), followed by the table header, followed
    // by another separator line.
    for (int i = 0; i < n_spaces_tot; i++) fprintf (pout, "-");
    fprintf (pout, "\n");
    fprintf (pout, "| %*s | %*s | %*s | %*s | %*s | %*s | %*s | %*s | %*s | %*s |\n",
	     n_func, headers[FUNC], 6, headers[MPI], n_calls, headers[CALLS],
             n_tot_send_bytes, headers[TOT_SEND_BYTES], n_tot_recv_bytes, headers[TOT_RECV_BYTES],
	     n_t_avg, headers[T_AVG],
	     n_t_min, headers[T_MIN], n_t_max, headers[T_MAX],
	     n_imba, headers[IMBA], n_t, headers[THIS_T]);
    for (int i = 0; i < n_spaces_tot; i++) fprintf (pout, "-");
    fprintf (pout, "\n");

    // Print all the display functions, but omit those without any calls.
    for (int i = 0; i < n_display_functions; i++) {

       // prepare the message size output
       char *send_unit_str;
       char *recv_unit_str;
       vftr_memory_unit(&(display_functions[i]->mpi_tot_send_bytes), &send_unit_str);
       vftr_memory_unit(&(display_functions[i]->mpi_tot_recv_bytes), &recv_unit_str);
	   
       if (display_functions[i]->n_calls > 0) {
	
	if (display_functions[i]->t_sync_avg > 0) {
	  // There are synchronization times for this function. We make space for the additional
	  // field "(xx.xx%)". Note that we need to subtract add_sync_spaces from the column widths.
          fprintf (pout, "| %*s | %6.2f | %*d | %*.2f %s | %*.2f %s | %*.5f(%5.2f%%) | %*.5f(%5.2f%%) | %*.5f(%5.2f%%) | %*.5f | %*.5f(%5.2f%%) |\n",
		   n_func, display_functions[i]->func_name,
 	  	   (display_functions[i]->this_mpi_time * 1e-6) / total_time * 100,
		   n_calls, display_functions[i]->n_calls,
                   n_tot_send_bytes-4, display_functions[i]->mpi_tot_send_bytes, send_unit_str,
                   n_tot_recv_bytes-4, display_functions[i]->mpi_tot_recv_bytes, recv_unit_str,
		   n_t_avg - add_sync_spaces, display_functions[i]->t_avg * 1e-6, (double)display_functions[i]->t_sync_avg / (double)display_functions[i]->t_avg * 100,
		   n_t_min - add_sync_spaces, display_functions[i]->t_min * 1e-6, (double)display_functions[i]->t_sync_min / (double)display_functions[i]->t_min * 100,
		   n_t_max - add_sync_spaces, display_functions[i]->t_max * 1e-6, (double)display_functions[i]->t_sync_max / (double)display_functions[i]->t_max * 100,
		   n_imba, display_functions[i]->imbalance,
		   n_t - add_sync_spaces, display_functions[i]->this_mpi_time * 1e-6, (double)display_functions[i]->this_sync_time / (double)display_functions[i]->this_mpi_time * 100);
	} else if (display_sync_time){
	   // This function does not have synchronization times, but others have. We take into
	   // account the synchronization fields "(xx.xx%)" of other functions by adding
	   // add_sync_spaces number of spaces. 
	   fprintf (pout, "| %*s | %6.2f | %*d | %*.2f %s | %*.2f %s | %*.5f         | %*.5f         | %*.5f         | %*.5f | %*.5f         |\n",
		    n_func, display_functions[i]->func_name,
		    (display_functions[i]->this_mpi_time * 1e-6) / total_time * 100,
		    n_calls, display_functions[i]->n_calls,
                    n_tot_send_bytes-4, display_functions[i]->mpi_tot_send_bytes, send_unit_str,
                    n_tot_recv_bytes-4, display_functions[i]->mpi_tot_recv_bytes, recv_unit_str,
 	            n_t_avg - add_sync_spaces, display_functions[i]->t_avg * 1e-6,
		    n_t_min - add_sync_spaces, display_functions[i]->t_min * 1e-6,
		    n_t_max - add_sync_spaces, display_functions[i]->t_max * 1e-6,
		    n_imba, display_functions[i]->imbalance,
		    n_t - add_sync_spaces, display_functions[i]->this_mpi_time * 1e-6);

	} else {
           // No display function has synchronization times, so only the absolute times are printed.
	   fprintf (pout, "| %*s | %6.2f | %*d | %*.2f %s | %*.2f %s | %*.5f | %*.5f | %*.5f | %*.5f | %*.5f |\n",
		    n_func, display_functions[i]->func_name,
		    (display_functions[i]->this_mpi_time * 1e-6) / total_time * 100,
		    n_calls, display_functions[i]->n_calls,
                    n_tot_send_bytes-4, display_functions[i]->mpi_tot_send_bytes, send_unit_str,
                    n_tot_recv_bytes-4, display_functions[i]->mpi_tot_recv_bytes, recv_unit_str,
 	            n_t_avg, display_functions[i]->t_avg * 1e-6,
		    n_t_min, display_functions[i]->t_min * 1e-6,
		    n_t_max, display_functions[i]->t_max * 1e-6,
		    n_imba, display_functions[i]->imbalance,
		    n_t, display_functions[i]->this_mpi_time * 1e-6);
       }
    }
  }
  //Print a final separator line.
  for (int i = 0; i < n_spaces_tot; i++) fprintf (pout, "-");
  fprintf (pout, "\n");

  if (vftr_environment.print_stack_profile->value) {
	// Next, we print the function stack trees. But first, we need to undo the sorting done before. 
	// This is because inside of vftr_print_function_stack, the run times of individual stacks are
	// gathered across all ranks. If the order of the display functions is not the same for all ranks,
	// which can happen for example when one of them is an I/O rank, the gathered values can correspond
	// to different functions.  	
	//
	qsort ((void*)display_functions, (size_t)n_display_functions,
	       sizeof (display_function_t *), vftr_compare_display_functions_iorig);


	if (vftr_mpirank == 0) {
	   mkdir ("html", 0777);
	   vftr_print_index_html (display_functions, n_display_functions);
	}
	PMPI_Barrier (MPI_COMM_WORLD);

  	for (int i = 0; i < n_display_functions; i++) {
		if (display_functions[i]->n_stack_indices == 0) {;
		   //print empty stack
		   //print epmty html
		} else {
		   stack_leaf_t *stack_tree = NULL;
		   double *imbalances = (double*) malloc (vftr_func_table_size * sizeof (double));
		   vftr_stack_compute_imbalances (&imbalances, display_functions[i]->n_stack_indices,
		   			       display_functions[i]->stack_indices);
		   vftr_create_stacktree (&stack_tree, display_functions[i]->n_stack_indices, display_functions[i]->stack_indices);
		   long long total_time = 0;
		   vftr_stack_get_total_time (stack_tree->origin, &total_time);
  		   vftr_print_function_stack (pout, vftr_mpirank, display_functions[i]->func_name, 
		   		      display_functions[i]->n_stack_indices,
		   		      display_functions[i]->n_func_indices,
  		   		      display_functions[i]->stack_indices,
		   		      display_functions[i]->func_indices,	
		   		      imbalances, total_time,
		   		      stack_tree);
		   vftr_print_html_output (NULL, display_functions, n_display_functions, i, stack_tree->origin,
					   imbalances, (double)total_time * 1e-6);
		   free (stack_tree);
		   free (imbalances);
	       }
	}
  }

  free (display_functions);
}

/**********************************************************************/

void display_selected_stacks (FILE *pout, char *display_function_names[], int n_display_functions) {

	//int n_indices, *stack_indices = NULL;	
	//for (int i = 0; i < n_display_functions; i++) {
	//	vftr_find_function_in_stack (display_function_names[i], &stack_indices, &n_indices, true);

	//	vftr_print_function_stack (pout, vftr_mpirank, display_function_names[i],
	//				   n_indices, 0, stack_indices, NULL);
	//	free (stack_indices);
	//}
	
}
/**********************************************************************/

void vftr_print_mpi_statistics (FILE *fp) {
    char *mpi_function_names[] = {"mpi_barrier", "mpi_bcast", "mpi_reduce",
			     "mpi_allreduce", "mpi_gather", "mpi_gatherv",
			     "mpi_allgather", "mpi_allgatherv",
			     "mpi_scatter", "mpi_scatterv",
			     "mpi_alltoall", "mpi_alltoallv", "mpi_alltoallw"};

    int n_mpi_functions = 13;
    vftr_print_function_statistics (fp, vftr_environment.mpi_show_sync_time->value,
				    mpi_function_names, n_mpi_functions);
}
#endif

/**********************************************************************/

void vftr_get_application_times (double time0, double *total_runtime, double *sampling_overhead_time,
			  	 double *mpi_overhead_time, double *total_overhead_time, double *application_time) {
   *total_runtime = time0 > 0 ? time0 * 1e-6 : vftr_get_runtime_usec() * 1e-6; 
   *sampling_overhead_time = vftr_overhead_usec * 1e-6;
   *total_overhead_time = *sampling_overhead_time;
#if defined(_MPI)
   *mpi_overhead_time = vftr_mpi_overhead_usec * 1e-6;
   *total_overhead_time = *sampling_overhead_time + *mpi_overhead_time;
#else
   *mpi_overhead_time = 0;
#endif
   *application_time = *total_runtime - *total_overhead_time;
}

/**********************************************************************/

void vftr_print_profile (FILE *pout, int *ntop, long long time0) {
    float pscale, ctime;
    double rtime;
    unsigned long long total_cycles, calls;
    evtcounter_t *evc0, *evc1, *evc;
    
    int n, k, fid;
    int offset, tableWidth;

    char fmtcalls[10], fmttime[10], fmttimeInc[10], fmtfid[10];

    function_t   **funcTable;

    long long *ectot = (long long *) malloc (vftr_n_hw_obs * sizeof (long long));
    for (int i = 0; i < vftr_n_hw_obs; i++) {
	ectot[i] = 0;
    }

    for (int i = 0; i < scenario_expr_n_vars; i++) {
	scenario_expr_counter_values[i] = 0.0;
    }

    evc1 = evc0 = vftr_get_counters();
    /* Find first event counter after scenario counters */

    if (!vftr_stackscount) return;

    funcTable = (function_t**) malloc (vftr_func_table_size * sizeof(function_t*));
    memcpy (funcTable, vftr_func_table, vftr_func_table_size * sizeof(function_t*));

    qsort( (void *)funcTable, (size_t)vftr_stackscount, sizeof( function_t *),
	   vftr_compare );    

    if (!vftr_profile_wanted)  return;

    total_cycles = 0;
    ctime = 0;
 
    /* Sum all cycles and counts */
    for (int i = 0; i < vftr_stackscount; i++) {
	if (funcTable[i] == NULL) continue;
	if (funcTable[i]->return_to && funcTable[i]->prof_current.calls) {
            profdata_t *prof_current  = &funcTable[i]->prof_current;
            profdata_t *prof_previous = &funcTable[i]->prof_previous;
	    total_cycles += prof_current->cycles - prof_previous->cycles;
            if (!prof_current->event_count || !prof_previous->event_count) continue;
	    for (int j = 0; j < scenario_expr_n_vars; j++) {
		scenario_expr_counter_values[j] += (double)(prof_current->event_count[j] - prof_previous->event_count[j]);
	    }
	}
    }
    double total_runtime, sampling_overhead_time, total_overhead_time, mpi_overhead_time, application_runtime;
    vftr_get_application_times (time0, &total_runtime, &sampling_overhead_time, &mpi_overhead_time, 
				&total_overhead_time, &application_runtime);
    rtime = application_runtime;

    /* Print profile info */

    fprintf(pout, "MPI size              %d\n", vftr_mpisize);
    fprintf(pout, "Total runtime:        %8.2f seconds\n", total_runtime);
    fprintf(pout, "Application time:     %8.2f seconds\n", application_runtime);
    	fprintf(pout, "Overhead:             %8.2f seconds (%.2f%%)\n",
            total_overhead_time, 100.0*total_overhead_time/total_runtime);
#ifdef _MPI
    fprintf(pout, "   Sampling overhead: %8.2f seconds (%.2f%%)\n",
            sampling_overhead_time, 100.0*sampling_overhead_time/total_runtime);
    fprintf(pout, "   MPI overhead:      %8.2f seconds (%.2f%%)\n",
            mpi_overhead_time, 100.0*mpi_overhead_time/total_runtime);
#endif

    /* Print overall info */
    if (vftr_events_enabled) {
	scenario_expr_evaluate_all (rtime, total_cycles);	
	scenario_expr_print_summary (pout);
    }

    /* Print all raw counter totals */
    fprintf( pout, "\nRaw counter totals\n"
            "------------------------------------------------------------\n"
            "%-37s : %20llu\n", 
            "Time Stamp Counter", total_cycles  );
    if (vftr_events_enabled) {
    	scenario_expr_print_raw_counters (pout);
    }

    fprintf( pout,
            "------------------------------------------------------------\n\n" );

    /* Save total counts */
    for (int i = 0; i < vftr_n_hw_obs; i++ ) {
	ectot[i];
    }

    pscale = 100. / rtime;
/* 
   Build a header with varying column widths, depending on the decimal places
   required. Column headers are truncated accordingly, but event names are
   printed in full on separate lines.
   Example:
                                      L2_RQSTS_PREFETCH_HIT
                                          L2_RQSTS_PREFETCH_MISS
                                                       L2_RQSTS_PREFETCHES
                                                             L2_RQSTS_REFERENCES
       Time             MFLOPS    L3
   Cal  (s)  %abs %cum Vecto Sca %Hit L2_ L2_RQSTS_PRE L2_RQ L2_RQS Functio Call stack
   --- ----- ---- ---- ----- --- ---- --- ------------ ----- ------ ------- --------------
   100 3.378 46.4 46.4  5692 237 97.4 378 373782558255 37825 378255 matmul3 <MAIN__<init
   100 1.302 17.8 64.3 15542   0 96.7 302 303024722472 30247 302472 matmul1 <MAIN__<init
   100 1.289 17.7 82.0 15566   0 96.7 289 282893779377 28937 289377 matmul2 <MAIN__<init
   100 1.265 17.3 99.4 15879   0 96.6 265 262651745174 26517 265174 matmul4 <MAIN__<init

*/

    fprintf (pout, "Runtime profile");
    if (vftr_mpisize > 1) {
        fprintf (pout, " for rank %d", vftr_mpirank);
    }
    if (vftr_environment.prof_truncate->value) {
	fprintf (pout, " (truncated)");
    }
    fprintf (pout, "\n");
    int n_indices = count_indices_to_evaluate (funcTable, application_runtime);
    int *indices = (int *)malloc (n_indices * sizeof(int));
    *ntop = n_indices;
    fill_indices_to_evaluate (funcTable, application_runtime, indices);

    /* Compute nr of decimal places needed */
    format_t *formats = (format_t *)malloc (sizeof(format_t));
    set_formats (funcTable, application_runtime, n_indices, indices, formats);
    formats->caller_name++; /* One more place to contain the asterisk marking missing event counts */
    if (vftr_events_enabled) {
    	set_evc_decipl (n_indices, vftr_n_hw_obs, evc1, evc);
    }

    /* Offset of first full event counter name header */

    offset = 1 + formats->n_calls + 1 + formats->excl_time + 1 + formats->incl_time + 1 + 5 + 5;
    if (evc0) offset += 5;

    if (vftr_events_enabled) {
    	offset += scenario_expr_get_table_width ();
    }

    /* The full event count headers start at "offset"
    ** First compute the full table width and print a
    ** "horizontal line"
    ** Skip if zero total count.
    */
    
    tableWidth = offset;
    if (vftr_events_enabled) {
    	int j = 0;
    	for (evc = evc1; evc; evc = evc->next ) {
    	    if (ectot[j++]) tableWidth += evc->decipl + 1;
    	}
    	tableWidth += formats->func_name + 1 
		   + formats->caller_name + 1 + formats->fid;
    }

    output_dashes_nextline (tableWidth, pout);

    /* Generic header line - 1 of 3 */ 

    n = 1 + formats->n_calls + 1
          + formats->excl_time + 1
          + formats->incl_time + 1 + 10;
    if (evc0) n += 5;

    if (vftr_events_enabled) {
        for (int i = 0; i < n; i++) {
		fputc (' ', pout);
	}
        scenario_expr_print_group (pout);
        fputs ("\n", pout);
    }

    /* Generic header line - 2 of 3 */ 

    fputs (" ", pout);
    output_header ("", formats->n_calls, pout);
    output_header ("Time[s]________________",
		   formats->excl_time + 1
		   + formats->incl_time, pout);
    n = 10;
    if (evc0) n += 5;
    if (vftr_events_enabled) {
        for (int i = 0; i < n; i++) {
		fputc (' ', pout);
	}
    	scenario_expr_print_subgroup (pout);
    }
    fputs ("\n ", pout);

    /* Generic header line - 3 of 3 */ 

    output_header ("Calls", formats->n_calls, pout);
    output_header ("Excl", formats->excl_time, pout);
    output_header ("Incl", formats->incl_time, pout);

    fputs ("%abs %cum ", pout);
    if (evc0) fputs ("%evc ", pout);

    if (vftr_events_enabled) {
        scenario_expr_print_header (pout);
    }

    if (vftr_events_enabled) {
    	int j = 0;
    	for (evc = evc1; evc; evc = evc->next) {
    	    if (ectot[j++]) {
    	        output_header (evc->name, evc->decipl, pout);
    	    }
    	}
    }
    output_header ("Function", formats->func_name, pout);
    output_header ("Caller", formats->caller_name, pout);
    output_header ("ID", formats->fid, pout);
    fputs ("\n", pout);

    /* Horizontal lines (collection of dashes) */
    fputs (" ", pout);

    output_dashes_nextline (tableWidth, pout);

    /* All headers printed at this point */
    /* Next: the numbers */

    sprintf (fmtcalls, "%%%dld ", formats->n_calls);
    sprintf (fmttimeInc, "%%%d.3f ", formats->incl_time);
    sprintf (fmttime, "%%%d.3f ", formats->excl_time);
    sprintf (fmtfid, "%%%dd", formats->fid);
    
    ctime = 0.;
    for (int i = 0; i < n_indices; i++) {
	int i_func = indices[i];
        profdata_t *prof_current   = &funcTable[i_func]->prof_current;
        profdata_t *prof_previous  = &funcTable[i_func]->prof_previous;

        calls  = prof_current->calls  - prof_previous->calls;
        fputc (' ', pout);
        fprintf (pout, fmtcalls, calls);

	float t_excl, t_incl, t_part;
	get_stack_times (prof_current, prof_previous, application_runtime, &t_excl, &t_incl, &t_part);
	rtime = t_excl;
	ctime += t_part;
	print_stack_time (pout, calls, fmttime, fmttimeInc, t_excl, t_incl, t_part, ctime);

        /* NOTE - counter info only printed for thread 0! */
	if (vftr_events_enabled) {
		fill_scenario_counter_values (scenario_expr_counter_values, scenario_expr_n_vars, 
			prof_current, prof_previous);

        	if (evc0) {
        	    long long reads = prof_current->ecreads- prof_previous->ecreads;
        	    double ratio = 100. * (double)reads / (double)calls;
        	    fprintf (pout, ratio < 99.95 ? "%4.1f " : "100. ", ratio);
        	}
        
		unsigned long long cycles = prof_current->cycles - prof_previous->cycles;
	    	scenario_expr_evaluate_all (rtime, cycles);
	    	//Formats should be set at this point
	    	scenario_expr_print_all_columns (pout);

        	int j = 0;
        	for (k = 0, evc = evc1; k < vftr_n_hw_obs; k++, evc = evc->next) {
        	    if (ectot[j++]) {
				fprintf (pout, evc->fmt, scenario_expr_counter_values[k]);
		    }
        	}
	}

	fprintf (pout, "%s", funcTable[i_func]->name);
        for (int j = strlen(funcTable[i_func]->name); j <= formats->func_name; j++) {
            fputc (' ', pout);
        }

	if (funcTable[i_func]->return_to) {
            fprintf (pout, "%s", funcTable[i_func]->return_to->name);
            for (int j = strlen(funcTable[i_func]->return_to->name); j <= formats->caller_name; j++) {
                fputc (' ', pout);
            }
        }

	fid = funcTable[i_func]->gid;
        fprintf (pout, fmtfid, fid);
        fprintf (pout, "\n");

    }
    
    output_dashes_nextline (tableWidth, pout);   
    fprintf( pout, "\n" );
    
    free (funcTable);
}

/**********************************************************************/

int vftr_filewrite_test_1 (FILE *fp_in, FILE *fp_out) {
	fprintf (fp_out, "Check the creation of log and vfd file name\n");
	int mpi_rank, mpi_size;
	mpi_rank = 0;
	mpi_size = 1;
	fprintf (fp_out, "logfile_name(%d, %d): %s\n", mpi_rank, mpi_size,
		 vftr_create_logfile_name(mpi_rank, mpi_size, "log"));
	mpi_rank = 11;
	mpi_size = 111;
	fprintf (fp_out, "logfile_name(%d, %d): %s\n", mpi_rank, mpi_size,
		 vftr_create_logfile_name(mpi_rank, mpi_size, "log"));
	fprintf (fp_out, "logfile_name(%d, %d): %s\n", mpi_rank, mpi_size,
		 vftr_create_logfile_name(mpi_rank, mpi_size, "vfd"));

	return 0;
}

/**********************************************************************/

int vftr_filewrite_test_2 (FILE *fp_in, FILE *fp_out) {
	int n;
	unsigned long long addrs [6];
	unsigned long long vftr_test_runtime = 0;
	function_t *func1 = vftr_new_function (NULL, "init", NULL, 0, false);
	function_t *func2 = vftr_new_function ((void*)addrs, "func2", func1, 0, false);
	function_t *func3 = vftr_new_function ((void*)(addrs + 1), "func3", func1, 0, false);	
	function_t *func4 = vftr_new_function ((void*)(addrs + 2), "func4", func3, 0, false);
	function_t *func5 = vftr_new_function ((void*)(addrs + 3), "func5", func2, 0, false);
	function_t *func6 = vftr_new_function ((void*)(addrs + 4), "func6", func2, 0, false);
	function_t *func7 = vftr_new_function ((void*)(addrs + 5), "func4", func6, 0, false);
	vftr_normalize_stacks();
	for (int i = 0; i < vftr_stackscount; i++) {
		vftr_func_table[i]->prof_current.calls = i + 1;
		vftr_func_table[i]->prof_current.cycles = 0;
		vftr_func_table[i]->prof_previous.cycles = 0;
		vftr_func_table[i]->prof_current.timeExcl = (long long)(i+1) * 100000;
		vftr_func_table[i]->prof_previous.timeExcl = (long long)(i+1) * 90000;
		vftr_func_table[i]->prof_current.timeIncl =
			2 * vftr_func_table[i]->prof_current.timeExcl;
		vftr_func_table[i]->prof_previous.timeIncl =
			2 * vftr_func_table[i]->prof_previous.timeExcl;
		vftr_test_runtime += vftr_func_table[i]->prof_current.timeExcl
				   - vftr_func_table[i]->prof_previous.timeExcl;
	}

	vftr_profile_wanted = true;
	vftr_mpisize = 1;
	vftr_overhead_usec = 0;
#ifdef _MPI
        vftr_mpi_overhead_usec = 0;
#endif
	vftr_print_profile (fp_out, &n, vftr_test_runtime);		
	return 0;
}

/**********************************************************************/

void vftr_memory_unit(double *value, char **unit) {
   int unit_idx = 0;
   while (*value > 1024.0) {
      unit_idx++;
      *value /= 1024.0;
   }

   switch (unit_idx) {
      case 0:
         *unit = "  B";
         break;
      case 1:
         *unit = "KiB";
         break;
      case 2:
         *unit = "MiB";
         break;
      case 3:
         *unit = "GiB";
         break;
      case 4:
         *unit = "TiB";
         break;
      case 5:
         *unit = "PiB";
         break;
      default:
         *unit = "  B";
         break;
   }
}

/**********************************************************************/

