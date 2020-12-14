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
#include "vftr_browse.h"
#include "vftr_sorting.h"

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

char *vftr_supported_mpi_function_names[] = {"mpi_accumulate",
					     "mpi_allgather", "mpi_allgatherv",
					     "mpi_allreduce",
				             "mpi_alltoall", "mpi_alltoallv", "mpi_alltoallw",
					     "mpi_barrier",
					     "mpi_bcast",
					     "mpi_bsend",
					     "mpi_exscan",
					     "mpi_fetch_and_op",
					     "mpi_finalize",
					     "mpi_gather", "mpi_gatherv",
					     "mpi_get", "mpi_get_accumulate",
					     "mpi_iallgather", "mpi_iallreduce",
					     "mpi_iallreduce",
					     "mpi_ialltoall", "mpi_ialltoallv", "mpi_ialltoallw",
					     "mpi_ibcast",
					     "mpi_ibsend",
					     "mpi_iexscan",
					     "mpi_igather", "mpi_igatherv",
					     "mpi_init",
					     "mpi_iprobe",
					     "mpi_irecv",
					     "mpi_ireduce", "mpi_ireduce_scatter", "mpi_ireduce_scatter_block",
					     "mpi_irsend",
					     "mpi_iscan",
					     "mpi_iscatter", "mpi_iscatterv",
					     "mpi_isend",
					     "mpi_issend",
					     "mpi_pcontrol",
					     "mpi_probe",
				             "mpi_put",
					     "mpi_raccumulate",
					     "mpi_recv",
					     "mpi_reduce", "mpi_reduce_scatter", "mpi_reduce_scatter_block",
					     "mpi_request_free",
					     "mpi_rget", "mpi_rget_accumulate",
					     "mpi_rput",
				             "mpi_rsend",
					     "mpi_scan",
					     "mpi_scatter", "mpi_scatterv",
				 	     "mpi_send",
					     "mpi_sendrecv",
					     "mpi_ssend",
					     "mpi_start", "mpi_startall",
					     "mpi_test", "mpi_testall", "mpi_testany", "mpi_testsome",
					     "mpi_wait", "mpi_waitall", "mpi_waitany", "mpi_waitsome"};

int vftr_n_supported_mpi_functions = 70;

char *vftr_mpi_collective_function_names[] = {"mpi_barrier", "mpi_bcast", "mpi_reduce",
			     "mpi_allreduce", "mpi_gather", "mpi_gatherv",
			     "mpi_allgather", "mpi_allgatherv",
			     "mpi_scatter", "mpi_scatterv",
			     "mpi_alltoall", "mpi_alltoallv", "mpi_alltoallw"};

int vftr_n_collective_mpi_functions = 13;

bool vftr_is_collective_mpi_function (char *func_name) {
   for (int i = 0; i < vftr_n_collective_mpi_functions; i++) { 
      if (!strcmp (func_name, vftr_mpi_collective_function_names[i])) return true;
   }
   return false;
}

void vftr_is_traceable_mpi_function (char *func_name, bool *is_mpi, bool *is_collective) {
   *is_mpi = false;
   *is_collective = false;
   if (!strncmp (func_name, "mpi_", 4)) {
      for (int i = 0; i < vftr_n_supported_mpi_functions; i++) {
         if (!strcmp (func_name, vftr_supported_mpi_function_names[i])) {
            *is_mpi = strcmp(func_name, "mpi_finalize");
	    break;
         }
      }
   }
   if (*is_mpi) {
      *is_collective = vftr_is_collective_mpi_function (func_name);
   }
   return;
}

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
	int task_digits = vftr_count_digits_int (mpi_size);
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
  	// vftr runtime
	fwrite (zerodouble, sizeof(double), 1, fp);
        // Three integers: function_samplecount, message_samplecount, stacks_count
	fwrite (zeroint, sizeof(unsigned int), 3, fp);
        // Two longs: stacks_offset, sample_offset 
	fwrite (zerolong, sizeof(long), 2, fp);
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
        fwrite (&runtime, sizeof(double), 1, vftr_vfd_file);
        fwrite (&vftr_function_samplecount, sizeof(unsigned int), 1, vftr_vfd_file);
        fwrite (&vftr_message_samplecount, sizeof(unsigned int), 1, vftr_vfd_file);
        fwrite (&vftr_stackscount, sizeof(unsigned int), 1, vftr_vfd_file);
        fwrite (&stackstable_offset, sizeof(long), 1, vftr_vfd_file);
        fwrite (&vftr_samples_offset, sizeof(long), 1, vftr_vfd_file);
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
    vftr_function_samplecount++;
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
                             long long tstart, long long tend,
                             int callingStackID) {
   
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
   fwrite(&callingStackID, sizeof(int), 1, vftr_vfd_file);

   vftr_message_samplecount++;
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
    
/**********************************************************************/
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
void vftr_output_column_header (char *header, int largest_column_length, FILE *fp) {
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

void vftr_print_stack_time (FILE *fp, int calls, double t_excl, double t_incl, double t_part, double t_cum, int *i_column, column_t *prof_columns) {

	double stime = calls ? t_excl : 0;
        fprintf (fp, prof_columns[(*i_column)++].format, stime);
	stime  = calls ? t_incl : 0;
        fprintf (fp, prof_columns[(*i_column)++].format, stime < 0. ? 0. : stime);
        fprintf (fp, prof_columns[(*i_column)++].format, t_part);
        fprintf (fp, prof_columns[(*i_column)++].format, t_cum);
}

/**********************************************************************/

void vftr_get_stack_times (profdata_t prof_current, profdata_t prof_previous, double runtime,
                      double *t_excl, double *t_incl, double *t_part) {
	long long timeExcl_usec = prof_current.timeExcl - prof_previous.timeExcl;
	long long timeIncl_usec = prof_current.timeIncl - prof_previous.timeIncl;
	*t_excl = timeExcl_usec * 1.0e-6;
	*t_incl = timeIncl_usec * 1.0e-6;
	*t_part = *t_excl * 100.0 / runtime;
}

/**********************************************************************/

void vftr_fill_func_indices_up_to_truncate (function_t **funcTable, double runtime, int *indices) {
	double cumulative_time = 0.;
    	double max_cumulative_time = 99.;
	double t_excl, t_incl, t_part;
	int j = 0;
	cumulative_time = 0.;
	for (int i = 0; i < vftr_stackscount; i++) {
		if (funcTable[i] == NULL) continue;
		profdata_t prof_current = funcTable[i]->prof_current;
		profdata_t prof_previous = funcTable[i]->prof_previous;
		/* If function has a caller and has been called */
		if (!(funcTable[i]->return_to && prof_current.calls)) continue;
		indices[j++] = i;
		vftr_get_stack_times (prof_current, prof_previous, runtime, &t_excl, &t_incl, &t_part);
		cumulative_time += t_part;
		if (vftr_environment.prof_truncate->value && cumulative_time > max_cumulative_time) break;
	}
}

/**********************************************************************/

int vftr_count_func_indices_up_to_truncate (function_t **funcTable, double runtime) {
	int n_indices = 0;
	double cumulative_time = 0.;
    	double max_cumulative_time = 99.;
	double t_excl, t_incl, t_part;
	for (int i = 0; i < vftr_stackscount; i++) {
		if (funcTable[i] == NULL) continue;
		profdata_t prof_current = funcTable[i]->prof_current;
		profdata_t prof_previous = funcTable[i]->prof_previous;
		/* If function has a caller and has been called */
		if (!(funcTable[i]->return_to && prof_current.calls)) continue;
		
		n_indices++;

		vftr_get_stack_times (prof_current, prof_previous, runtime, &t_excl, &t_incl, &t_part);
		cumulative_time += t_part;
		if (vftr_environment.prof_truncate->value && cumulative_time > max_cumulative_time) break;
	}
	return n_indices;
}

/**********************************************************************/

void vftr_fill_scenario_counter_values (double *val, int n_vars, profdata_t prof_current, profdata_t prof_previous) {
	memset (vftr_scenario_expr_counter_values, 0., sizeof (double) * vftr_scenario_expr_n_vars);
	if (prof_current.event_count) {
		for (int i = 0; i < n_vars; i++) {
			val[i] += (double)prof_current.event_count[i];
		}
	}
        if (prof_previous.event_count) {
		for (int i = 0; i < n_vars; i++) {
			val[i] -= (double)prof_previous.event_count[i];
		}
	}
}

/**********************************************************************/

void vftr_prof_column_init (const char *name, char *group_header, int n_decimal_places, int type, column_t *c) {
	c->header = strdup (name);
	c->group_header = group_header != NULL ? strdup (group_header) : NULL;
	c->n_chars = strlen(name);
	c->n_decimal_places = n_decimal_places;
	c->type = type;
}

void vftr_prof_column_set_n_chars (void *value, column_t *c) {
	int n;
	int *i;
	double *f;
        char *ch;
	switch (c->type) {
	   case COL_INT:
	      i = (int*)value;	
	      n = vftr_count_digits_int (*i);
	      break;
	   case COL_DOUBLE:
	      f = (double *)value;
	      // For double values, we need to add the decimal places and the decimal point itself.
	      n = vftr_count_digits_double (*f) + c->n_decimal_places + 1;
	      break;
	   case COL_CHAR:
	      ch = (char*)value;
   	      n = strlen(ch);
	      break;
	}
	if (n > c->n_chars) c->n_chars = n;
}

void vftr_prof_column_set_format (column_t *c) {
	switch (c->type) {
	   case COL_INT:
	      sprintf (c->format, " %%%dd ", c->n_chars);
	      break;
	   case COL_DOUBLE:
              sprintf (c->format, " %%%d.%df ", c->n_chars, c->n_decimal_places);
	      break;
	   case COL_CHAR:
	      sprintf (c->format, " %%%ds ", c->n_chars);
	}
}

/**********************************************************************/

void vftr_set_summary_column_formats (bool print_mpi, int n_display_funcs, display_function_t **display_functions, column_t **columns) {
    const char *headers[10] = {"Function", "%MPI", "Calls",
                              "Total send ", "Total recv.",
			      "Avg. time [s]", "Min. time [s]", "Max. time [s]",
			      "Imbalance", "This rank [s]"};
    enum column_ids {FUNC, MPI, CALLS, TOT_SEND_BYTES, TOT_RECV_BYTES, T_AVG, T_MIN, T_MAX, IMBA, THIS_T};

    int i_column = 0;
    vftr_prof_column_init (headers[FUNC], NULL, 0, COL_CHAR, &(*columns)[i_column++]);
    if (print_mpi) vftr_prof_column_init (headers[MPI], NULL, 2, COL_DOUBLE, &(*columns)[i_column++]);
    vftr_prof_column_init (headers[CALLS], NULL, 0, COL_INT, &(*columns)[i_column++]);
    if (print_mpi) {
       vftr_prof_column_init (headers[TOT_SEND_BYTES], NULL, 2, COL_DOUBLE, &(*columns)[i_column++]);
       vftr_prof_column_init (headers[TOT_RECV_BYTES], NULL, 2, COL_DOUBLE, &(*columns)[i_column++]);
    }
    vftr_prof_column_init (headers[T_AVG], NULL, 2, COL_DOUBLE, &(*columns)[i_column++]);
    vftr_prof_column_init (headers[T_MIN], NULL, 2, COL_DOUBLE, &(*columns)[i_column++]);
    vftr_prof_column_init (headers[T_MAX], NULL, 2, COL_DOUBLE, &(*columns)[i_column++]);
    vftr_prof_column_init (headers[IMBA], NULL, 2, COL_DOUBLE, &(*columns)[i_column++]);
    vftr_prof_column_init (headers[THIS_T], NULL, 2, COL_DOUBLE, &(*columns)[i_column++]);
  
    for (int i = 0; i < n_display_funcs; i++) {
       i_column = 0;
       vftr_prof_column_set_n_chars (&display_functions[i]->func_name, &(*columns)[i_column++]);
       // Set n_chars for %MPI
       vftr_prof_column_set_n_chars (&display_functions[i]->n_calls, &(*columns)[i_column++]);
       if (print_mpi) {
	  vftr_prof_column_set_n_chars (&display_functions[i]->mpi_tot_send_bytes, &(*columns)[i_column++]);
	  vftr_prof_column_set_n_chars (&display_functions[i]->mpi_tot_recv_bytes, &(*columns)[i_column++]);
       } 
       double t = display_functions[i]->t_avg * 1e-6;
       vftr_prof_column_set_n_chars (&t, &(*columns)[i_column++]);
       t = display_functions[i]->t_min * 1e-6;
       vftr_prof_column_set_n_chars (&t, &(*columns)[i_column++]);
       t = display_functions[i]->t_max * 1e-6;
       vftr_prof_column_set_n_chars (&t, &(*columns)[i_column++]);
       vftr_prof_column_set_n_chars (&display_functions[i]->imbalance, &(*columns)[i_column++]);
       vftr_prof_column_set_n_chars (&display_functions[i]->this_mpi_time, &(*columns)[i_column++]);
    }
}

/**********************************************************************/

void vftr_set_proftab_column_formats (function_t **funcTable,
	double runtime, double sampling_overhead_time, 
	int n_funcs, int *func_indices, column_t **columns) {
	int i_column = 0;
        vftr_prof_column_init ("Calls", NULL, 0, COL_INT, &(*columns)[i_column++]);
        vftr_prof_column_init ("t_excl[s]", NULL, 3, COL_DOUBLE, &(*columns)[i_column++]);
        vftr_prof_column_init ("t_incl[s]", NULL, 3, COL_DOUBLE, &(*columns)[i_column++]);
        vftr_prof_column_init ("%abs", NULL, 1, COL_DOUBLE, &(*columns)[i_column++]);
        vftr_prof_column_init ("%cum", NULL, 1, COL_DOUBLE, &(*columns)[i_column++]);

        if (vftr_environment.show_overhead->value) {
	   vftr_prof_column_init ("t_ovhd[s]", NULL, 3, COL_DOUBLE, &(*columns)[i_column++]);
	   vftr_prof_column_init ("%ovhd", NULL, 1, COL_DOUBLE, &(*columns)[i_column++]);
	   vftr_prof_column_init ("t_ovhd / t_excl", NULL, 1, COL_DOUBLE, &(*columns)[i_column++]);
        }

	if (vftr_events_enabled) {
		char header_with_unit[80];
		for (int i = 0; i < vftr_scenario_expr_n_formulas; i++) {
		   if (vftr_scenario_expr_format[i].header == NULL) {
			sprintf (header_with_unit, "HWC N/A");
		   } else {
		      int n1 = strlen(vftr_scenario_expr_format[i].header);
		      if (vftr_scenario_expr_format[i].unit) {
		           int n2 = strlen(vftr_scenario_expr_format[i].unit); 
		           snprintf (header_with_unit, n1 + n2 + 4, "%s [%s]", vftr_scenario_expr_format[i].header, vftr_scenario_expr_format[i].unit);
		      } else {
		           snprintf (header_with_unit, n1 + 1, "%s", vftr_scenario_expr_format[i].header);
		      }
		   }
		   vftr_prof_column_init (header_with_unit, NULL, vftr_scenario_expr_format[i].decimal_places,
		           		  COL_DOUBLE, &(*columns)[i_column++]);
		} 
 	}

        // Set scenario columns here
        vftr_prof_column_init ("Function", NULL, 0, COL_CHAR, &(*columns)[i_column++]);
        vftr_prof_column_init ("Caller", NULL, 0, COL_CHAR, &(*columns)[i_column++]);
        vftr_prof_column_init ("ID", NULL, 0, COL_INT, &(*columns)[i_column++]);
        double t_cum = 0;
        for (int i = 0; i < n_funcs; i++) {
            int i_func = func_indices[i];
            profdata_t prof_current = funcTable[i_func]->prof_current;
            profdata_t prof_previous = funcTable[i_func]->prof_previous;
            double t_excl, t_incl, t_part, t_overhead;
            vftr_get_stack_times (prof_current, prof_previous, runtime, &t_excl, &t_incl, &t_part);
            t_cum += t_part;
            t_overhead = (double)funcTable[i_func]->overhead * 1e-6;
            int n_calls = prof_current.calls - prof_previous.calls;
            i_column = 0;
            vftr_prof_column_set_n_chars (&n_calls, &(*columns)[i_column++]);
            vftr_prof_column_set_n_chars (&t_excl, &(*columns)[i_column++]);
            vftr_prof_column_set_n_chars (&t_incl, &(*columns)[i_column++]);
            vftr_prof_column_set_n_chars (&t_part, &(*columns)[i_column++]);
            vftr_prof_column_set_n_chars (&t_cum, &(*columns)[i_column++]);

            if (vftr_environment.show_overhead->value) {
                vftr_prof_column_set_n_chars (&t_overhead, &(*columns)[i_column++]);
                double rel = sampling_overhead_time > 0.0 ? t_overhead / sampling_overhead_time * 100.0 : 0.0;
                vftr_prof_column_set_n_chars (&rel, &(*columns)[i_column++]);
                rel = t_excl > 0.0 ? t_overhead / t_excl : 0.0;
                vftr_prof_column_set_n_chars (&rel, &(*columns)[i_column++]);
            }

	    if (vftr_events_enabled) {
		vftr_fill_scenario_counter_values (vftr_scenario_expr_counter_values,
		          vftr_scenario_expr_n_vars, prof_current, prof_previous);
		unsigned long long cycles = prof_current.cycles - prof_previous.cycles;
		vftr_scenario_expr_evaluate_all (t_excl, cycles);
		for (int j = 0; j < vftr_scenario_expr_n_formulas; j++) {
		   double tmp = vftr_scenario_expr_formulas[j].value;
		   vftr_prof_column_set_n_chars (&tmp, &(*columns)[i_column++]);
	   	}
	    }

            vftr_prof_column_set_n_chars (funcTable[i_func]->name, &(*columns)[i_column++]);
            vftr_prof_column_set_n_chars (funcTable[i_func]->return_to->name, &(*columns)[i_column++]);
	}
	columns[0]->n_chars++;
}

/**********************************************************************/

int vftr_get_tablewidth_from_columns (column_t *columns, int n_columns, bool use_separators) {
   // Initialize tablewidth to the number of spaces (two per column), then
   // add the individual column widths to it.
   int tw = 2 * n_columns;
   // If separator lines (|) are used, add their number (n_columns + 1)
   tw += n_columns + 1;
   for (int i = 0; i < n_columns; i++) {
      tw += columns[i].n_chars;
   }
   return tw;
}

/**********************************************************************/

void vftr_summary_print_header (FILE *fp, column_t *columns, int table_width, bool print_mpi) {
   enum column_ids {FUNC, MPI, CALLS, TOT_SEND_BYTES, TOT_RECV_BYTES, T_AVG, T_MIN, T_MAX, IMBA, THIS_T};
   for (int i = 0; i < table_width; i++) fprintf (fp, "-");
   fprintf (fp, "\n");
   if (print_mpi) {
      fprintf (fp, "| %*s | %*s | %*s | %*s | %*s | %*s | %*s | %*s | %*s | %*s |\n",
	       columns[FUNC].n_chars, columns[FUNC].header,
               columns[MPI].n_chars, columns[MPI].header,
	       columns[CALLS].n_chars, columns[CALLS].header,
 	       columns[TOT_SEND_BYTES].n_chars, columns[TOT_SEND_BYTES].header,
	       columns[TOT_RECV_BYTES].n_chars, columns[TOT_RECV_BYTES].header,
	       columns[T_AVG].n_chars, columns[T_AVG].header,
	       columns[T_MIN].n_chars, columns[T_MIN].header,
	       columns[T_MAX].n_chars, columns[T_MAX].header,
	       columns[IMBA].n_chars, columns[IMBA].header,
	       columns[THIS_T].n_chars, columns[THIS_T].header);
   } else {
      fprintf (fp, "| %*s | %*s | %*s | %*s | %*s | %*s | %*s |\n",
	       columns[FUNC].n_chars, columns[FUNC].header,
	       columns[CALLS].n_chars, columns[CALLS].header,
	       columns[T_AVG].n_chars, columns[T_AVG].header,
	       columns[T_MIN].n_chars, columns[T_MIN].header,
	       columns[T_MAX].n_chars, columns[T_MAX].header,
	       columns[IMBA].n_chars, columns[IMBA].header,
	       columns[THIS_T].n_chars, columns[THIS_T].header);
   }
   for (int i = 0; i < table_width; i++) fprintf (fp, "-");
   fprintf (fp, "\n");
}

/**********************************************************************/

void vftr_proftab_print_header (FILE *fp, column_t *columns) {
	// NOTE: The evaluation order of function arguments is not strictly defined.
	// Therefore, do not increment an array index if it is used more than
	// once in the function call, like func(a[i], b[i++]);
	int i;
	for (i = 0; i < 5; i++) {
	   fprintf (fp, " %*s ", columns[i].n_chars, columns[i].header);
	}

	if (vftr_environment.show_overhead->value) {
	   fprintf (fp, " %*s ", columns[i].n_chars, columns[i].header);
	   i++; // See above
	   fprintf (fp, " %*s ", columns[i].n_chars, columns[i].header);
	   i++;
	   fprintf (fp, " %*s ", columns[i].n_chars, columns[i].header);
	   i++;
	}

	if (vftr_events_enabled) {
	   for (int j = 0; j < vftr_scenario_expr_n_formulas; j++) {
	      fprintf (fp, " %*s ", columns[i].n_chars, columns[i].header);
	      i++;
	   }
   	}
			
	fprintf (fp, " %*s ", columns[i].n_chars, columns[i].header);
	i++;
	fprintf (fp, " %*s ", columns[i].n_chars, columns[i].header);
	i++;
	fprintf (fp, " %*s ", columns[i].n_chars, columns[i].header);
	i++;
	fprintf (fp, "\n");
}

/**********************************************************************/

#ifdef _MPI
double vftr_compute_mpi_imbalance (long long *all_times, double t_avg) {
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

void vftr_evaluate_display_function (char *func_name, display_function_t **display_func,
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
    (*display_func)->properly_terminated = true;
    for (int i = 0; i < n_func_indices; i++) {
	(*display_func)->this_mpi_time += vftr_func_table[func_indices[i]]->prof_current.timeIncl;
	if (n_func_indices_sync > 0) (*display_func)->this_sync_time += vftr_func_table[func_indices_sync[i]]->prof_current.timeIncl;
	(*display_func)->n_calls += vftr_func_table[func_indices[i]]->prof_current.calls;
	(*display_func)->mpi_tot_send_bytes += vftr_func_table[func_indices[i]]->prof_current.mpi_tot_send_bytes;
	(*display_func)->mpi_tot_recv_bytes += vftr_func_table[func_indices[i]]->prof_current.mpi_tot_recv_bytes;
	(*display_func)->properly_terminated &= !vftr_func_table[func_indices[i]]->open;
    }
    if (!(*display_func)->properly_terminated) return;
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
       (*display_func)->imbalance = vftr_compute_mpi_imbalance (all_times, (*display_func)->t_avg);
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
		if (display_functions[i]->n_calls > 0 && display_functions[i]->properly_terminated) {
			n = strlen (display_functions[i]->func_name);
			if (n > *n_func_max) *n_func_max = n;
			n = vftr_count_digits_int (display_functions[i]->n_calls);
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

//void vftr_print_function_statistics (FILE *fp_log, bool display_sync_time, 
//				     char *display_function_names[], int n_display_functions) {
void vftr_print_function_statistics (FILE *fp_log, bool display_sync_time, int *func_indices, int n_indices) {

    display_function_t **display_functions =
			//(display_function_t**) malloc (n_display_functions * sizeof(display_function_t*));
			(display_function_t**) malloc (n_indices * sizeof(display_function_t*));


    //for (int i = 0; i < n_display_functions; i++) {
    int i_disp_f = 0;
    //printf ("Count display functions\n");
    for (int i = 0; i < n_indices; i++) {
        bool name_already_there = false;
	int i_func = func_indices[i];
	for (int j = 0; j < i_disp_f; j++) {
	   if (!strcmp(display_functions[j]->func_name, vftr_gStackinfo[i_func].name)) {
	     name_already_there = true;
	     break;
           }
        }
	if (name_already_there) continue;
	display_functions[i_disp_f] = (display_function_t*) malloc (sizeof(display_function_t));
	display_functions[i_disp_f]->func_name = strdup(vftr_gStackinfo[i_func].name);
	//vftr_find_function_in_stack (display_functions[i]->func_name, &stack_indices, &n_stack_indices, true);
	vftr_is_traceable_mpi_function (display_functions[i_disp_f]->func_name,
				        &(display_functions[i_disp_f]->is_mpi), &(display_functions[i_disp_f]->is_collective_mpi));
        display_functions[i_disp_f]->i_orig = i_disp_f;
	i_disp_f++;
    }

    int n_display_funcs = i_disp_f;
    //printf ("DO REALLOC: %d\n", vftr_mpirank);
    //display_functions = (display_function_t **) realloc(display_functions, n_display_funcs * sizeof(display_function_t*));
    //printf ("HAVE REALLOCED: %d\n", vftr_mpirank);
    
    double total_time = 0;
    //for (int i = 0; i < n_display_functions; i++) {
    //char **display_function_names = (char**)malloc (n_indices * sizeof(char*));
    //for (int i = 0; i < n_indices; i++) {
    for (int i = 0; i < n_display_funcs; i++) {
       //vftr_evaluate_display_function (display_function_names[i], &(display_functions[i]), display_sync_time);
       vftr_evaluate_display_function (display_functions[i]->func_name, &(display_functions[i]), display_sync_time);
       if (!display_functions[i]->properly_terminated) continue;
       total_time += display_functions[i]->this_mpi_time * 1e-6;
    }

    //qsort ((void*)display_functions, (size_t)n_indices,
    qsort ((void*)display_functions, (size_t)n_display_funcs,
	    sizeof (display_function_t *), vftr_compare_display_functions_tavg);


    fprintf (fp_log, "Total time spent in MPI for rank %d: %lf s\n", vftr_mpirank, total_time);
    fprintf (fp_log, "Imbalance computed as: max (T - T_avg)\n");

    // Most of this code deals with the determination of the column widths.
    // vftr_get_display_width loops over all display_functions and determines
    // the maximum number of digits or characters required to display the
    // corresponding column. 
    int n_func, n_calls, n_t_avg, n_t_min, n_t_max, n_imba, n_t;
    //vftr_get_display_width (display_functions, n_indices, 5,
    vftr_get_display_width (display_functions, n_display_funcs, 5,
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

    bool print_mpi_columns = true;
    int n_columns = print_mpi_columns ? 10 : 7;

    column_t *columns = (column_t*) malloc (n_columns * sizeof(column_t));
    vftr_set_summary_column_formats (print_mpi_columns, n_display_funcs, display_functions, &columns);
    for (int i = 0; i < n_columns; i++) {
       vftr_prof_column_set_format (&(columns[i]));
    }
   
    int table_width = vftr_get_tablewidth_from_columns (columns, n_columns, true); 
    if (vftr_mpirank == 0) {
      vftr_summary_print_header (stdout, columns, table_width, print_mpi_columns);
    }

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
    for (int i = 0; i < n_spaces_tot; i++) fprintf (fp_log, "-");
    fprintf (fp_log, "\n");
    fprintf (fp_log, "| %*s | %*s | %*s | %*s | %*s | %*s | %*s | %*s | %*s | %*s |\n",
	     n_func, headers[FUNC], 6, headers[MPI], n_calls, headers[CALLS],
             n_tot_send_bytes, headers[TOT_SEND_BYTES], n_tot_recv_bytes, headers[TOT_RECV_BYTES],
	     n_t_avg, headers[T_AVG],
	     n_t_min, headers[T_MIN], n_t_max, headers[T_MAX],
	     n_imba, headers[IMBA], n_t, headers[THIS_T]);
    for (int i = 0; i < n_spaces_tot; i++) fprintf (fp_log, "-");
    fprintf (fp_log, "\n");

    // Print all the display functions, but omit those without any calls.
    //for (int i = 0; i < n_indices; i++) {
    for (int i = 0; i < n_display_funcs; i++) {

       if (display_functions[i]->n_calls > 0 && display_functions[i]->properly_terminated) {
         // prepare the message size output
         char *send_unit_str;
         char *recv_unit_str;
         vftr_memory_unit(&(display_functions[i]->mpi_tot_send_bytes), &send_unit_str);
         vftr_memory_unit(&(display_functions[i]->mpi_tot_recv_bytes), &recv_unit_str);
	   
	
	if (display_functions[i]->t_sync_avg > 0) {
	  // There are synchronization times for this function. We make space for the additional
	  // field "(xx.xx%)". Note that we need to subtract add_sync_spaces from the column widths.
          fprintf (fp_log, "| %*s | %6.2f | %*d | %*.2f %s | %*.2f %s | %*.5f(%5.2f%%) | %*.5f(%5.2f%%) | %*.5f(%5.2f%%) | %*.5f | %*.5f(%5.2f%%) |\n",
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
	   fprintf (fp_log, "| %*s | %6.2f | %*d | %*.2f %s | %*.2f %s | %*.5f         | %*.5f         | %*.5f         | %*.5f | %*.5f         |\n",
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
	   fprintf (fp_log, "| %*s | %6.2f | %*d | %*.2f %s | %*.2f %s | %*.5f | %*.5f | %*.5f | %*.5f | %*.5f |\n",
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
  for (int i = 0; i < n_spaces_tot; i++) fprintf (fp_log, "-");
  fprintf (fp_log, "\n");

  if (vftr_environment.print_stack_profile->value) {
	// Next, we print the function stack trees. But first, we need to undo the sorting done before. 
	// This is because inside of vftr_print_function_stack, the run times of individual stacks are
	// gathered across all ranks. If the order of the display functions is not the same for all ranks,
	// which can happen for example when one of them is an I/O rank, the gathered values can correspond
	// to different functions.  	
	//
	//qsort ((void*)display_functions, (size_t)n_indices,
	qsort ((void*)display_functions, (size_t)n_display_funcs,
	       sizeof (display_function_t *), vftr_compare_display_functions_iorig);


	if (vftr_environment.create_html->value) {
	   if (vftr_mpirank == 0) {
	      vftr_browse_print_index_html (vftr_mpi_collective_function_names, vftr_n_collective_mpi_functions);
	   }
        }

  	//for (int i = 0; i < n_indices; i++) {
  	for (int i = 0; i < n_display_funcs; i++) {
                if (!display_functions[i]->properly_terminated) continue;
		if (display_functions[i]->n_stack_indices == 0) {;
	 	   if (vftr_environment.create_html->value) {
		      vftr_browse_print_stacktree_page (NULL, true, vftr_mpi_collective_function_names,
		         				vftr_n_collective_mpi_functions, i, NULL, NULL, 0.0, 0, 0);
          	   }
		} else {
		   stack_leaf_t *stack_tree = NULL;
		   double *imbalances = (double*) malloc (vftr_func_table_size * sizeof (double));
		   vftr_stack_compute_imbalances (imbalances, display_functions[i]->n_stack_indices,
		   			       display_functions[i]->stack_indices);
		   vftr_create_stacktree (&stack_tree, display_functions[i]->n_stack_indices, display_functions[i]->stack_indices);
		   long long total_time = 0;
		   vftr_stack_get_total_time (stack_tree->origin, &total_time);

		   double t_max, imba_max;
		   int n_calls_max, n_spaces_max, n_chars_max;
		   vftr_scan_stacktree (stack_tree, display_functions[i]->n_stack_indices, imbalances,
					&t_max, &n_calls_max, &imba_max, &n_spaces_max, &n_chars_max);
  		   vftr_print_function_stack (fp_log, display_functions[i]->func_name, 
		   		              display_functions[i]->n_stack_indices,
		   		              imbalances, total_time,
					      t_max, n_calls_max, imba_max, n_spaces_max,
					      stack_tree);
		   if (vftr_environment.create_html->value) {
		      vftr_browse_print_stacktree_page (NULL, false, vftr_mpi_collective_function_names,
						      vftr_n_collective_mpi_functions, i, stack_tree->origin,
					              imbalances, (double)total_time * 1e-6, n_chars_max,
						      display_functions[i]->n_stack_indices);
		   }
		   free (stack_tree);
		   free (imbalances);
	       }
	}
  }

  free (display_functions);
}

/**********************************************************************/

void vftr_print_mpi_statistics (FILE *fp) {
       vftr_stackid_list_init();
       for (int i = 0; i < vftr_gStackscount; i++) {
           if (vftr_pattern_match (vftr_environment.print_stack_profile->value, vftr_gStackinfo[i].name)) {
              vftr_stackid_list_add (i);
           }
       }
    vftr_stackid_list_print (stdout);

    vftr_print_function_statistics (fp, vftr_environment.mpi_show_sync_time->value,
				    //vftr_mpi_collective_function_names, vftr_n_collective_mpi_functions);
				    print_stackid_list, n_print_stackids);
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

void vftr_print_profile_summary (FILE *fp_log, function_t **func_table, double total_runtime, double application_runtime,
				 double total_overhead_time, double sampling_overhead_time, double mpi_overhead_time) {

    fprintf(fp_log, "MPI size              %d\n", vftr_mpisize);
    fprintf(fp_log, "Total runtime:        %8.2f seconds\n", total_runtime);
    fprintf(fp_log, "Application time:     %8.2f seconds\n", application_runtime);
    fprintf(fp_log, "Overhead:             %8.2f seconds (%.2f%%)\n",
            total_overhead_time, 100.0*total_overhead_time/total_runtime);
#ifdef _MPI
    fprintf(fp_log, "   Sampling overhead: %8.2f seconds (%.2f%%)\n",
            sampling_overhead_time, 100.0*sampling_overhead_time/total_runtime);
    fprintf(fp_log, "   MPI overhead:      %8.2f seconds (%.2f%%)\n",
            mpi_overhead_time, 100.0*mpi_overhead_time/total_runtime);
#endif

    if (vftr_events_enabled) {
	unsigned long long total_cycles = 0;

        for (int i = 0; i < vftr_stackscount; i++) {
    	   if (func_table[i] == NULL) continue;
    	   if (func_table[i]->return_to && func_table[i]->prof_current.calls) {
              profdata_t *prof_current  = &func_table[i]->prof_current;
              profdata_t *prof_previous = &func_table[i]->prof_previous;
    	      total_cycles += prof_current->cycles - prof_previous->cycles;
              if (!prof_current->event_count || !prof_previous->event_count) continue;
    	      for (int j = 0; j < vftr_scenario_expr_n_vars; j++) {
    	          vftr_scenario_expr_counter_values[j] += (double)(prof_current->event_count[j] - prof_previous->event_count[j]);
    	      }
    	   }
        }

	vftr_scenario_expr_evaluate_all (application_runtime, total_cycles);	
	vftr_scenario_expr_print_summary (fp_log);

        fprintf (fp_log, "\nRaw counter totals\n"
            "------------------------------------------------------------\n"
            "%-37s : %20llu\n", 
            "Time Stamp Counter", total_cycles);
    	vftr_scenario_expr_print_raw_counters (fp_log);

    }
    fprintf (fp_log, "------------------------------------------------------------\n\n");
}

/**********************************************************************/

void vftr_compute_line_content (function_t *this_func, double application_runtime, double sampling_overhead_time,
			        int *n_calls, double *t_excl, double *t_incl, double *t_part, double *t_overhead) {
   profdata_t prof_current = this_func->prof_current;
   profdata_t prof_previous = this_func->prof_previous;
   *n_calls = (int)(prof_current.calls - prof_previous.calls);

   vftr_get_stack_times (prof_current, prof_previous, application_runtime, t_excl, t_incl, t_part);
   if (vftr_environment.show_overhead->value) {
      *t_overhead = this_func->overhead * 1e-6;
   } else {
      *t_overhead = 0.0;
   }

   if (vftr_events_enabled) {
   	vftr_fill_scenario_counter_values (vftr_scenario_expr_counter_values, vftr_scenario_expr_n_vars, 
   		prof_current, prof_previous);
       	vftr_scenario_expr_evaluate_all (*t_excl, prof_current.cycles - prof_previous.cycles);
   }
  
   
}

void vftr_print_profile_line (FILE *fp_log, int stack_id, double sampling_overhead_time,
			      int n_calls, double t_excl, double t_incl, double t_part, double t_cum, double t_overhead,
			      char *func_name, char *caller_name, column_t *prof_columns) {
   int i_column = 0;
   fprintf (fp_log, prof_columns[i_column++].format, n_calls);
   vftr_print_stack_time (fp_log, n_calls, t_excl, t_incl, t_part, t_cum, &i_column, prof_columns);

   if (vftr_environment.show_overhead->value) {
      fprintf (fp_log, prof_columns[i_column++].format, t_overhead);
      fprintf (fp_log, prof_columns[i_column++].format, t_overhead / sampling_overhead_time * 100.0);
      fprintf (fp_log, prof_columns[i_column++].format, t_excl > 0 ? t_overhead / t_excl : 0.0);
   }
   
   if (vftr_events_enabled) {
       	for (int i = 0; i < vftr_scenario_expr_n_formulas; i++) {
   	   fprintf (fp_log, prof_columns[i_column++].format, vftr_scenario_expr_formulas[i].value);
   	}
   }
   
   fprintf (fp_log, prof_columns[i_column++].format, func_name);
   if (caller_name) {
       fprintf (fp_log, prof_columns[i_column++].format, caller_name);
   }
   
   fprintf (fp_log, prof_columns[i_column++].format, stack_id);
   fprintf (fp_log, "\n");
}

/**********************************************************************/

void vftr_print_profile (FILE *fp_log, int *n_func_indices, long long time0) {
    unsigned long long calls;
    
    int table_width;

    FILE *f_html;

    if (!vftr_stackscount) return;
    if (!vftr_profile_wanted)  return;

    if (vftr_environment.create_html->value) {
       vftr_browse_create_directory ();
       f_html = vftr_browse_init_profile_table ();
    }

    function_t **func_table;

    for (int i = 0; i < vftr_scenario_expr_n_vars; i++) {
	vftr_scenario_expr_counter_values[i] = 0.0;
    }

    func_table = (function_t**) malloc (vftr_func_table_size * sizeof(function_t*));
    // Create a local copy of the global function table to sort it.
    memcpy (func_table, vftr_func_table, vftr_func_table_size * sizeof(function_t*));

    qsort ((void *)func_table, (size_t)vftr_stackscount, sizeof (function_t *), vftr_get_profile_compare_function());

    double total_runtime, sampling_overhead_time, total_overhead_time, mpi_overhead_time, application_runtime;
    vftr_get_application_times (time0, &total_runtime, &sampling_overhead_time, &mpi_overhead_time, 
				&total_overhead_time, &application_runtime);

    vftr_print_profile_summary (fp_log, func_table, total_runtime, application_runtime,
				total_overhead_time, sampling_overhead_time, mpi_overhead_time);

    fprintf (fp_log, "Runtime profile");
    if (vftr_mpisize > 1) {
        fprintf (fp_log, " for rank %d", vftr_mpirank);
    }
    if (vftr_environment.prof_truncate->value) {
	fprintf (fp_log, " (truncated)");
    }
    fprintf (fp_log, ":\n\n");

    *n_func_indices = vftr_count_func_indices_up_to_truncate (func_table, application_runtime);
    int *func_indices = (int *)malloc (*n_func_indices * sizeof(int));
    vftr_fill_func_indices_up_to_truncate (func_table, application_runtime, func_indices);

    // Number of columns. Default: nCalls, exclusive & inclusive time, %abs, %cum,
    // function & caller name and stack ID (i.e. 8 columns). 
    int n_columns = 8;
    // Add one column for each hardware counter.
    n_columns += vftr_scenario_expr_n_formulas;
    // If function overhead is displayed, add three more columns.
    if (vftr_environment.show_overhead->value) n_columns += 3;

    int i_column = 0;
    column_t *prof_columns = (column_t*) malloc (n_columns * sizeof(column_t));
    vftr_set_proftab_column_formats (func_table, application_runtime, sampling_overhead_time, *n_func_indices, func_indices, &prof_columns);

    for (int i = 0; i < n_columns; i++) {
	vftr_prof_column_set_format (&(prof_columns[i]));
    }

    table_width = vftr_get_tablewidth_from_columns (prof_columns, n_columns, false);

    if (vftr_environment.create_html->value) {
       vftr_browse_create_profile_header (f_html);
    }

    vftr_proftab_print_header (fp_log, prof_columns);
    vftr_print_dashes (fp_log, table_width);

    // All headers printed at this point
    // Next: the numbers

    double cumulative_time = 0.;
    for (int i = 0; i < *n_func_indices; i++) {
       int i_func = func_indices[i];
       int n_calls;
       double t_excl, t_incl, t_part, t_overhead;
       vftr_compute_line_content (func_table[i_func], application_runtime, sampling_overhead_time,
				  &n_calls, &t_excl, &t_incl, &t_part, &t_overhead);
       cumulative_time += t_part;
       vftr_print_profile_line (fp_log, func_table[i_func]->gid, sampling_overhead_time,
			        n_calls, t_excl, t_incl, t_part, cumulative_time, t_overhead,
				func_table[i_func]->name, func_table[i_func]->return_to->name, prof_columns);

       if (vftr_environment.create_html->value) {
	  vftr_browse_print_table_line (f_html, func_table[i_func]->gid, sampling_overhead_time,
					n_calls, t_excl, t_incl, t_part, cumulative_time, t_overhead,
				        func_table[i_func]->name, func_table[i_func]->return_to->name, prof_columns);
       }
    }

    if (vftr_environment.create_html->value) vftr_browse_finalize_table(f_html);
    
    vftr_print_dashes (fp_log, table_width);
    fprintf (fp_log, "\n");
    
    free (func_table);
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

void vftr_time_unit (double *value, char **unit, bool for_html) {
   int unit_idx = 0;
   // Explicitly treat zero. Otherwise, the loop below will be indefinite.
   if (*value == 0.0) {
      *unit = "s";
      return;
   }

   while (*value < 1.0) {
	unit_idx++;
	*value *= 1000;
   }

   switch (unit_idx) {
      case 0:
	  *unit = "s";
	  break;
      case 1:
	  *unit = "ms";
	  break;
      case 2:
	  *unit = for_html ? "&#956s" : "mus";
	  break;
      case 3:
	  *unit = "ns";
          break;
   }
}

/**********************************************************************/

