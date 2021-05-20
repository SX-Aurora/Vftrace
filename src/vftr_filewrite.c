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
#include "vftr_mallinfo.h"
#include "vftr_allocate.h"

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

void vftr_is_traceable_mpi_function (char *func_name, bool *is_mpi) {
   *is_mpi = false;
   char *func_name_low = vftr_to_lowercase(func_name);
   if (!strncmp (func_name_low, "mpi_", 4)) {
      for (int i = 0; i < vftr_n_supported_mpi_functions; i++) {
         if (!strcmp (func_name_low, vftr_supported_mpi_function_names[i])) {
            *is_mpi = strcmp(func_name_low, "mpi_finalize");
	    break;
         }
      }
   }
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
		  if ((s = rindex (program_path, '/'))) {
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

void vftr_finalize_vfd_file (long long finalize_time) {
  long stackstable_offset = ftell (vftr_vfd_file);
  vftr_write_stacks_vfd (vftr_vfd_file, 0, vftr_froots);

  double runtime = finalize_time * 1.0e-6;

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

/**********************************************************************/

void vftr_write_to_vfd(long long runtime, profdata_t *prof_current, profdata_t *prof_previous, int stack_id, unsigned int sid) {
    fwrite (&sid, sizeof(unsigned int), 1, vftr_vfd_file);
    fwrite (&stack_id, sizeof(int), 1, vftr_vfd_file);
    fwrite (&runtime, sizeof(long long), 1, vftr_vfd_file);

    vftr_write_observables_to_vfd (prof_current, prof_previous, vftr_vfd_file);

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

    function_t   **func_table;

    if (!vftr_stackscount)return;
    func_table = vftr_func_table;

    ec = (unsigned long long *) malloc (vftr_n_hw_obs * sizeof(long long));
    for (int j = 0; j < vftr_n_hw_obs; j++) {
	ec[j] = 0;
    }

    total_cycles = 0;
 
    /* Sum all cycles and counts */
    for (int i = 0; i < vftr_stackscount; i++) {
	if (func_table[i] && func_table[i]->return_to && func_table[i]->prof_current.calls) {
            profdata_t *prof_current = &func_table[i]->prof_current;
	    total_cycles += prof_current->cycles;
            if (!prof_current->event_count) continue;
            for (int j = 0; j < vftr_n_hw_obs; j++) {
                ec[j] += prof_current->event_count[j];
	    }
            if (vftr_memtrace) {
               ec[vftr_n_hw_obs-1] += prof_current->event_count[vftr_n_hw_obs-1];
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
        if (!func_table[i]) continue;
        profdata_t *prof_current  = &func_table[i]->prof_current;
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
        vftr_prof_column_print (fp, prof_columns[(*i_column)++], &stime, NULL, NULL);
	stime  = calls ? t_incl : 0;
        vftr_prof_column_print (fp, prof_columns[(*i_column)++], &stime, NULL, NULL);
        vftr_prof_column_print (fp, prof_columns[(*i_column)++], &t_part, NULL, NULL);
        vftr_prof_column_print (fp, prof_columns[(*i_column)++], &t_cum, NULL, NULL);
}

/**********************************************************************/

void vftr_get_stack_times (profdata_t prof_current, profdata_t prof_previous, long long *t_excl_usec, long long *t_incl_usec) {
	*t_excl_usec = prof_current.time_excl - prof_previous.time_excl;
	*t_incl_usec = prof_current.time_incl - prof_previous.time_incl;
}

/**********************************************************************/

void vftr_fill_func_indices_up_to_truncate (function_t **func_table, long long runtime_usec, int *indices) {
	long long cumulative_time_usec = 0;
	long long t_excl, t_incl;
	int j = 0;
	for (int i = 0; i < vftr_stackscount; i++) {
		if (func_table[i] == NULL) continue;
		profdata_t prof_current = func_table[i]->prof_current;
		profdata_t prof_previous = func_table[i]->prof_previous;
		/* If function has a caller and has been called */
		if (!(func_table[i]->return_to && prof_current.calls)) continue;
		indices[j++] = i;
		vftr_get_stack_times (prof_current, prof_previous, &t_excl, &t_incl);
		cumulative_time_usec += t_excl;
		double t_cum = (double)cumulative_time_usec / (double)runtime_usec * 100.0;
		if (vftr_environment.prof_truncate->value && t_cum > vftr_environment.prof_truncate_cutoff->value) break;
	}
}

/**********************************************************************/

int vftr_count_func_indices_up_to_truncate (function_t **func_table, long long runtime_usec) {
	int n_indices = 0;
	long long cumulative_time_usec = 0;
	long long t_excl, t_incl;
	for (int i = 0; i < vftr_stackscount; i++) {
		if (func_table[i] == NULL) continue;
		profdata_t prof_current = func_table[i]->prof_current;
		profdata_t prof_previous = func_table[i]->prof_previous;
		/* If function has a caller and has been called */
		if (!(func_table[i]->return_to && prof_current.calls)) continue;
		
		n_indices++;

		vftr_get_stack_times (prof_current, prof_previous, &t_excl, &t_incl);
		cumulative_time_usec += t_excl;
		double t_cum = (double)cumulative_time_usec / (double)runtime_usec * 100.0;
		if (vftr_environment.prof_truncate->value && t_cum > vftr_environment.prof_truncate_cutoff->value) break;
	}
	return n_indices;
}

/**********************************************************************/

void vftr_fill_scenario_counter_values (double *val, int n_vars, profdata_t prof_current, profdata_t prof_previous) {
	memset (vftr_scenario_expr_counter_values, 0., sizeof (double) * vftr_scenario_expr_n_vars);
	//if (prof_current.event_count) {
	//	for (int i = 0; i < n_vars; i++) {
	//		val[i] += (double)prof_current.event_count[i];
	//	}
	//}
        //if (prof_previous.event_count) {
	//	for (int i = 0; i < n_vars; i++) {
	//		val[i] -= (double)prof_previous.event_count[i];
	//	}
	//}
	//
	if (prof_current.event_count) {
           vftr_scenario_expr_counter_values[0] += (double)prof_current.event_count[vftr_n_hw_obs-1];
        }
	if (prof_previous.event_count) vftr_scenario_expr_counter_values[0] -= (double)prof_previous.event_count[vftr_n_hw_obs-1];
        vftr_scenario_expr_counter_values[0] = vftr_scenario_expr_counter_values[0] / prof_current.calls / 1024 / 1024;
    
}

/**********************************************************************/

void vftr_prof_column_init (const char *name, char *group_header, int n_decimal_places, int col_type, int sep_type, column_t *c) {
   c->header = strdup (name);
   c->group_header = group_header != NULL ? strdup (group_header) : NULL;
   c->n_chars = strlen(name);
   c->n_chars_opt_1 = 0;
   c->n_chars_opt_2 = 0;
   c->n_decimal_places = n_decimal_places;
   c->col_type = col_type;
   c->separator_type = sep_type;
}

/**********************************************************************/

void vftr_prof_column_print_status (FILE *fp, column_t c) {
   fprintf (fp, "Column type: "); 
   switch (c.col_type) {
      case COL_INT:
        fprintf (fp, "Integer\n");
	break;
      case COL_DOUBLE:
        fprintf (fp, "Double\n");
        break;
      case COL_CHAR_RIGHT:
        fprintf (fp, "Right-aligned String\n");
        break;
      case COL_CHAR_LEFT:
        fprintf (fp, "Left-aligned String\n");
        break;
      case COL_MEM:
        fprintf (fp, "Data Size\n");
        break;
      case COL_TIME:
        fprintf (fp, "Time\n");
	break;
   }
   fprintf (fp, "Title: %s\n", c.header);
   fprintf (fp, "n_chars: %d\n", c.n_chars);
   fprintf (fp, "n_chars_opt_1: %d\n", c.n_chars_opt_1);
   fprintf (fp, "n_chars_opt_2: %d\n", c.n_chars_opt_2);
   fprintf (fp, "n_decimal_placse: %d\n", c.n_decimal_places);
   fprintf (fp, "Separator: ");
   if (c.separator_type == SEP_LAST || c.separator_type == SEP_MID) {
      fprintf (fp, "|\n");
   } else {
      fprintf (fp, " \n");
   }
}
/**********************************************************************/

//void vftr_prof_column_set_n_chars (void *value_1, void *value_2, column_t *c, int *stat) {
void vftr_prof_column_set_n_chars (void *value, void *opt_1, void *opt_2, column_t *c, int *stat) {
	int n;
	int *i;
	double *f;
        char *ch;
	switch (c->col_type) {
	   case COL_INT:
	      i = (int*)value;
	      n = vftr_count_digits_int (*i);
	      break;
	   case COL_DOUBLE:
	      f = (double *)value;
	      // For double values, we need to add the decimal places and the decimal point itself.
	      n = vftr_count_digits_double (*f) + c->n_decimal_places + 1;
	      break;
	   case COL_CHAR_RIGHT:
           case COL_CHAR_LEFT:
	      ch = (char*)value;
   	      n = strlen(ch);
	      break;
	   case COL_MEM:
              // Floating-point value + memory unit
              f = (double *)value;
  	      n = strlen(vftr_memory_unit_string (*(double*)value, c->n_decimal_places));    
              break;
           case COL_TIME:
              f = (double *)value;
	      n = vftr_count_digits_double (*f) + c->n_decimal_places + 1;
	      if (opt_1) {
                 double *f2 = (double *)opt_1;
	         if (*f2 > 0) {
	            int n2 = vftr_count_digits_double (*f2 / *f * 100) + 5; // Two decimal places, a comma and two brackets
		    if (n2 > c->n_chars_opt_1) c->n_chars_opt_1 = n2;
		    n += n2;
	         }
              }
	      if (opt_2) {
                 int *f3 = (int *)opt_2;
                 int n3 = vftr_count_digits_int (*f3) + 2; // Two brackets
                 if (n3 > c->n_chars_opt_2) c->n_chars_opt_2 = n3;
                 n += n3;
              }
              if (opt_1 != NULL && opt_2 != NULL) n++; // Add the space in between
	      break;
           default:
              *stat = -1;
	}
	if (n > c->n_chars) c->n_chars = n;
	*stat = n;
}

/**********************************************************************/

void vftr_prof_column_print (FILE *fp, column_t c, void *value, void *opt_1, void *opt_2) {
   if (c.separator_type == SEP_MID || c.separator_type == SEP_LAST) fprintf (fp, "|");
   switch (c.col_type) {
      case COL_INT:
         fprintf (fp, " %*d ", c.n_chars, *(int*)value);
         break;
      case COL_DOUBLE:
         fprintf (fp, " %*.*f ", c.n_chars, c.n_decimal_places, *(double*)value);
	 break;
      case COL_CHAR_RIGHT:
         fprintf (fp, " %*s ", c.n_chars, (char*)value);
	 break;
      case COL_CHAR_LEFT:
         fprintf (fp, " %s ", (char*)value);
	 break;
      case COL_MEM:
         fprintf (fp, " %*s ", c.n_chars, vftr_memory_unit_string (*(double*)value, c.n_decimal_places));
	 break;
      case COL_TIME:
         if (opt_1 != NULL && opt_2 != NULL && c.n_chars_opt_1 > 0 && c.n_chars_opt_2 > 0) {
            fprintf (fp, " %*.*f(%*.2f) (%*d) ", c.n_chars - c.n_chars_opt_1 - c.n_chars_opt_2 - 1, c.n_decimal_places,
	   	     *(double*)value, c.n_chars_opt_1 - 2, *(double*)opt_1 / *(double*)value * 100.0,
                     c.n_chars_opt_2 - 2, *(int*)opt_2);
         } else if (opt_1 != NULL && c.n_chars_opt_1 > 0) {
            fprintf (fp, " %*.*f(%*.2f) ", c.n_chars - c.n_chars_opt_1 - 1, c.n_decimal_places,
	   	     *(double*)value, c.n_chars_opt_1 - 2, *(double*)opt_1 / *(double*)value * 100.0);
	 } else if (opt_2 != NULL && c.n_chars_opt_2 > 0) {
            fprintf (fp, " %*.*f(%*d) ", c.n_chars - c.n_chars_opt_2 - 1, c.n_decimal_places, *(double*)value,
                     c.n_chars_opt_2 - 2, *(int*)opt_2);
	 } else {
            fprintf (fp, " %*.*f ", c.n_chars, c.n_decimal_places, *(double*)value);
	 }
   }
   if (c.separator_type == SEP_LAST) fprintf (fp, "|");
}

/**********************************************************************/

void vftr_set_summary_column_formats (bool print_mpi, int n_display_funcs, display_function_t **display_functions, column_t **columns, double total_time) {
    const char *headers[11] = {"Function", "Stack ID", "%MPI", "Calls",
                              "Total send ", "Total recv.",
			      "Avg. time [s]", "Min. time [s]", "Max. time [s]",
			      "Imbalance [%]", "This rank [s]"};
    enum column_ids {FUNC, STACK_ID, MPI, CALLS, TOT_SEND_BYTES, TOT_RECV_BYTES, T_AVG, T_MIN, T_MAX, IMBA, THIS_T};

    int i_column = 0;
    vftr_prof_column_init (headers[FUNC], NULL, 0, COL_CHAR_RIGHT, SEP_MID, &(*columns)[i_column++]);
    if (vftr_environment.all_mpi_summary->value) vftr_prof_column_init (headers[STACK_ID], NULL, 0, COL_INT, SEP_MID, &(*columns)[i_column++]);
    if (print_mpi) vftr_prof_column_init (headers[MPI], NULL, 2, COL_DOUBLE, SEP_MID, &(*columns)[i_column++]);
    vftr_prof_column_init (headers[CALLS], NULL, 0, COL_INT, SEP_MID, &(*columns)[i_column++]);
    if (print_mpi) {
       vftr_prof_column_init (headers[TOT_SEND_BYTES], NULL, 0, COL_MEM, SEP_MID, &(*columns)[i_column++]);
       vftr_prof_column_init (headers[TOT_RECV_BYTES], NULL, 0, COL_MEM, SEP_MID, &(*columns)[i_column++]);
    }
    vftr_prof_column_init (headers[T_AVG], NULL, 2, COL_TIME, SEP_MID, &(*columns)[i_column++]);
    vftr_prof_column_init (headers[T_MIN], NULL, 2, COL_TIME, SEP_MID, &(*columns)[i_column++]);
    vftr_prof_column_init (headers[T_MAX], NULL, 2, COL_TIME, SEP_MID, &(*columns)[i_column++]);
    vftr_prof_column_init (headers[IMBA], NULL, 2, COL_DOUBLE, SEP_MID, &(*columns)[i_column++]);
    vftr_prof_column_init (headers[THIS_T], NULL, 2, COL_TIME, SEP_LAST, &(*columns)[i_column++]);
  
    int stat;
    for (int i = 0; i < n_display_funcs; i++) {
       if (!display_functions[i]->properly_terminated) continue;       
       i_column = 0;
       vftr_prof_column_set_n_chars (display_functions[i]->func_name, NULL, NULL, &(*columns)[i_column++], &stat);
       if (vftr_environment.all_mpi_summary->value) {
          int stack_id = display_functions[i]->stack_id;
          vftr_prof_column_set_n_chars (&stack_id, NULL, NULL, &(*columns)[i_column++], &stat);
       }
       if (print_mpi) {
          double tmp = display_functions[i]->this_mpi_time * 1e-6 / total_time * 100;
	  vftr_prof_column_set_n_chars (&tmp, NULL, NULL, &(*columns)[i_column++], &stat);
       }
       vftr_prof_column_set_n_chars (&display_functions[i]->n_calls, NULL, NULL, &(*columns)[i_column++], &stat);
       if (print_mpi) {
	  vftr_prof_column_set_n_chars (&display_functions[i]->mpi_tot_send_bytes, NULL, NULL, &(*columns)[i_column++], &stat);  
	  vftr_prof_column_set_n_chars (&display_functions[i]->mpi_tot_recv_bytes, NULL, NULL, &(*columns)[i_column++], &stat);
       } 
       double t = display_functions[i]->t_avg * 1e-6;
       double t2 = display_functions[i]->t_sync_avg * 1e-6;
       bool has_sync = t2 > 0;
       vftr_prof_column_set_n_chars (&t, has_sync ? &t2 : NULL, NULL, &(*columns)[i_column++], &stat);
       t = display_functions[i]->t_min * 1e-6;
       t2 = display_functions[i]->t_sync_min * 1e-6;
       int n = display_functions[i]->rank_min;
       vftr_prof_column_set_n_chars (&t, has_sync ? &t2 : NULL, &n, &(*columns)[i_column++], &stat);
       t = display_functions[i]->t_max * 1e-6;
       t2 = display_functions[i]->t_sync_max * 1e-6;
       n = display_functions[i]->rank_max;
       vftr_prof_column_set_n_chars (&t, has_sync ? &t2 : NULL, &n, &(*columns)[i_column++], &stat);
       vftr_prof_column_set_n_chars (&display_functions[i]->imbalance, NULL, NULL, &(*columns)[i_column++], &stat);
       t = display_functions[i]->this_mpi_time * 1e-6;
       t2 = display_functions[i]->this_sync_time * 1e-6;
       vftr_prof_column_set_n_chars (&t, has_sync ? &t2 : NULL, NULL, &(*columns)[i_column++], &stat);
    }
}

/**********************************************************************/

void vftr_set_proftab_column_formats (function_t **func_table,
	long long runtime_usec, double sampling_overhead_time, 
	int n_funcs, int *func_indices, column_t *columns) {
	int i_column = 0;
        vftr_prof_column_init ("Calls", NULL, 0, COL_INT, SEP_NONE, &(columns)[i_column++]);
        vftr_prof_column_init ("t_excl[s]", NULL, 3, COL_DOUBLE, SEP_NONE, &(columns)[i_column++]);
        vftr_prof_column_init ("t_incl[s]", NULL, 3, COL_DOUBLE, SEP_NONE, &(columns)[i_column++]);
        vftr_prof_column_init ("%abs", NULL, 1, COL_DOUBLE, SEP_NONE, &(columns)[i_column++]);
        vftr_prof_column_init ("%cum", NULL, 1, COL_DOUBLE, SEP_NONE, &(columns)[i_column++]);

        if (vftr_environment.show_overhead->value) {
	   vftr_prof_column_init ("t_ovhd[s]", NULL, 3, COL_DOUBLE, SEP_NONE, &(columns)[i_column++]);
	   vftr_prof_column_init ("%ovhd", NULL, 1, COL_DOUBLE, SEP_NONE, &(columns)[i_column++]);
	   vftr_prof_column_init ("t_ovhd / t_excl", NULL, 1, COL_DOUBLE, SEP_NONE, &(columns)[i_column++]);
        }

	if (vftr_events_enabled) {
           vftr_prof_column_init ("MEMORY", NULL, 2, COL_DOUBLE, SEP_NONE, &(columns)[i_column++]);
        //    printf ("BLUB BLUB\n");
	//	char header_with_unit[80];
	//	for (int i = 0; i < vftr_scenario_expr_n_formulas; i++) {
	//	   if (vftr_scenario_expr_format[i].header == NULL) {
	//		sprintf (header_with_unit, "HWC N/A");
	//	   } else {
	//	      int n1 = strlen(vftr_scenario_expr_format[i].header);
	//	      if (vftr_scenario_expr_format[i].unit) {
	//	           int n2 = strlen(vftr_scenario_expr_format[i].unit); 
	//	           snprintf (header_with_unit, n1 + n2 + 4, "%s [%s]", vftr_scenario_expr_format[i].header, vftr_scenario_expr_format[i].unit);
	//	      } else {
	//	           snprintf (header_with_unit, n1 + 1, "%s", vftr_scenario_expr_format[i].header);
	//	      }
	//	   }
	//	   vftr_prof_column_init (header_with_unit, NULL, vftr_scenario_expr_format[i].decimal_places,
	//	           		  COL_DOUBLE, SEP_NONE, &(columns)[i_column++]);
	//	} 
 	}

        if (vftr_max_allocated_fields > 0) vftr_prof_column_init ("Max mem", NULL, 2, COL_MEM, SEP_NONE, &(columns)[i_column++]);

        vftr_prof_column_init ("Function", NULL, 0, COL_CHAR_RIGHT, SEP_NONE, &(columns)[i_column++]);
        vftr_prof_column_init ("Caller", NULL, 0, COL_CHAR_RIGHT, SEP_NONE, &(columns)[i_column++]);
        vftr_prof_column_init ("ID", NULL, 0, COL_INT, SEP_NONE, &(columns)[i_column++]);
        if (vftr_environment.show_stacks_in_profile->value) {
           vftr_prof_column_init ("Stack", NULL, 0, COL_CHAR_LEFT, SEP_NONE, &(columns)[i_column++]);
        }
        int stat;
        long long t_sum = 0;
        for (int i = 0; i < n_funcs; i++) {
            int i_func = func_indices[i];
	    if (func_table[i_func]->open) continue;
            profdata_t prof_current = func_table[i_func]->prof_current;
            profdata_t prof_previous = func_table[i_func]->prof_previous;
            long long t_excl, t_incl;
            vftr_get_stack_times (prof_current, prof_previous, &t_excl, &t_incl);
     	    t_sum += t_excl;
            int n_calls = prof_current.calls - prof_previous.calls;
            i_column = 0;
            vftr_prof_column_set_n_chars (&n_calls, NULL, NULL, &(columns)[i_column++], &stat);
            vftr_prof_column_set_n_chars (&t_excl, NULL, NULL, &(columns)[i_column++], &stat);
            vftr_prof_column_set_n_chars (&t_incl, NULL, NULL, &(columns)[i_column++], &stat);
	    double t = (double)t_excl / (double)runtime_usec * 100;
            vftr_prof_column_set_n_chars (&t, NULL, NULL, &(columns)[i_column++], &stat);
	    t = (double)t_sum / (double)runtime_usec * 100;
            vftr_prof_column_set_n_chars (&t, NULL, NULL, &(columns)[i_column++], &stat);

            if (vftr_environment.show_overhead->value) {
                double t_overhead = func_table[i_func]->overhead * 1e-6;
                vftr_prof_column_set_n_chars (&t_overhead, NULL, NULL, &(columns)[i_column++], &stat);
                double rel = sampling_overhead_time > 0.0 ? t_overhead / sampling_overhead_time * 99.99 : 0.0;
                vftr_prof_column_set_n_chars (&rel, NULL, NULL, &(columns)[i_column++], &stat);
                rel = t_excl > 0.0 ? t_overhead / t_excl : 0.0;
                vftr_prof_column_set_n_chars (&rel, NULL, NULL, &(columns)[i_column++], &stat);
            }

	    if (vftr_events_enabled) {
		vftr_fill_scenario_counter_values (vftr_scenario_expr_counter_values,
		          vftr_scenario_expr_n_vars, prof_current, prof_previous);
		unsigned long long cycles = prof_current.cycles - prof_previous.cycles;
		//vftr_scenario_expr_evaluate_all (t_excl * 1e-6, cycles);
                //printf ("NFORMULAS: %d\n", vftr_scenario_expr_n_formulas);
		//for (int j = 0; j < vftr_scenario_expr_n_formulas; j++) {
		   //double tmp = vftr_scenario_expr_formulas[j].value;
		   //vftr_prof_column_set_n_chars (&tmp, NULL, NULL, &(columns)[i_column++], &stat);
		   vftr_prof_column_set_n_chars (&vftr_scenario_expr_counter_values, NULL, NULL, &(columns)[i_column++], &stat);
	   	//}
	    }

            if (vftr_max_allocated_fields > 0) {
               double mem_max = (double)vftr_allocate_get_max_memory_for_stackid (func_table[i_func]->id);
	       vftr_prof_column_set_n_chars (&mem_max, NULL, NULL, &(columns)[i_column++], &stat);
            }

            vftr_prof_column_set_n_chars (func_table[i_func]->name, NULL, NULL, &(columns)[i_column++], &stat);
            if (func_table[i_func]->return_to != NULL) {
              vftr_prof_column_set_n_chars (func_table[i_func]->return_to->name, NULL, NULL, &(columns)[i_column++], &stat);
            } else {
              vftr_prof_column_set_n_chars ("-/-", NULL, NULL, &(columns)[i_column++], &stat);
            }
            int global_id = func_table[i_func]->gid;
            vftr_prof_column_set_n_chars (&global_id, NULL, NULL, &(columns)[i_column++], &stat);
            if (vftr_environment.show_stacks_in_profile->value) {
               vftr_prof_column_set_n_chars (vftr_global_stack_strings[global_id].s, NULL, NULL, &(columns)[i_column++], &stat);
	    }
	}
	columns[0].n_chars++;
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
	       columns[0].n_chars, columns[0].header,
               columns[1].n_chars, columns[1].header,
	       columns[2].n_chars, columns[2].header,
 	       columns[3].n_chars, columns[3].header,
	       columns[4].n_chars, columns[4].header,
	       columns[5].n_chars, columns[5].header,
	       columns[6].n_chars, columns[6].header,
	       columns[7].n_chars, columns[7].header,
	       columns[8].n_chars, columns[8].header,
	       columns[9].n_chars, columns[9].header);
   } else if (vftr_environment.all_mpi_summary->value) {
      fprintf (fp, "| %*s | %*s | %*s | %*s | %*s | %*s | %*s | %*s |\n",
	       columns[0].n_chars, columns[0].header,
	       columns[1].n_chars, columns[1].header,
	       columns[2].n_chars, columns[2].header,
	       columns[3].n_chars, columns[3].header,
	       columns[4].n_chars, columns[4].header,
	       columns[5].n_chars, columns[5].header,
	       columns[6].n_chars, columns[6].header,
	       columns[7].n_chars, columns[7].header);
   } else {
      fprintf (fp, "| %*s | %*s | %*s | %*s | %*s | %*s | %*s |\n",
	       columns[0].n_chars, columns[0].header,
	       columns[1].n_chars, columns[1].header,
	       columns[2].n_chars, columns[2].header,
	       columns[3].n_chars, columns[3].header,
	       columns[4].n_chars, columns[4].header,
	       columns[5].n_chars, columns[5].header,
	       columns[6].n_chars, columns[6].header);
   }
   for (int i = 0; i < table_width; i++) fprintf (fp, "-");
   fprintf (fp, "\n");
   fprintf (fp, "malloc_info overhead: %lf\n", (double)vftr_mallinfo_ovhd * 1e-6);
   fprintf (fp, "malloc_info parsing overhead: %lf\n", (double)vftr_mallinfo_post_ovhd * 1e-6);
}

/**********************************************************************/

void vftr_summary_print_line (FILE *fp, display_function_t *displ_f, column_t *columns, double total_time, bool print_mpi) {
   int i_column = 0;
   vftr_prof_column_print (fp, columns[i_column++], displ_f->func_name, NULL, NULL);
   if (vftr_environment.all_mpi_summary->value) vftr_prof_column_print (fp, columns[i_column++], &displ_f->stack_id, NULL, NULL);
   double t, t2;
   if (print_mpi) {
      t = displ_f->this_mpi_time * 1e-6 / total_time * 100;
      vftr_prof_column_print (fp, columns[i_column++], &t, NULL, NULL);
   }
   vftr_prof_column_print (fp, columns[i_column++], &displ_f->n_calls, NULL, NULL);
   if (print_mpi) {
      vftr_prof_column_print (fp, columns[i_column++], &displ_f->mpi_tot_send_bytes, NULL, NULL);
      vftr_prof_column_print (fp, columns[i_column++], &displ_f->mpi_tot_recv_bytes, NULL, NULL);
   }
   t = displ_f->t_avg * 1e-6;
   t2 = displ_f->t_sync_avg * 1e-6;
   bool has_sync = t2 > 0;
   vftr_prof_column_print (fp, columns[i_column++], &t, has_sync ? &t2 : NULL, NULL);
   t = displ_f->t_min * 1e-6;
   t2 = displ_f->t_sync_min * 1e-6;
   int n = displ_f->rank_min;
   vftr_prof_column_print (fp, columns[i_column++], &t, has_sync ? &t2 : NULL, &n);
   t = displ_f->t_max * 1e-6;
   t2 = displ_f->t_sync_max * 1e-6;
   n = displ_f->rank_max;
   vftr_prof_column_print (fp, columns[i_column++], &t, has_sync ? &t2 : NULL, &n);
   vftr_prof_column_print (fp, columns[i_column++], &displ_f->imbalance, NULL, NULL);
   t = displ_f->this_mpi_time * 1e-6;
   t2 = displ_f->this_sync_time * 1e-6;
   vftr_prof_column_print (fp, columns[i_column++], &t, has_sync ? &t2 : NULL, NULL);
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
	   //for (int j = 0; j < vftr_scenario_expr_n_formulas; j++) {
	   for (int j = 0; j < vftr_n_hw_obs; j++) {
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
        if (vftr_environment.show_stacks_in_profile->value) {
          fprintf (fp, " %*s ", columns[i].n_chars, columns[i].header);
	  i++;
        }
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
#endif

void vftr_evaluate_display_function (char *func_name, display_function_t **display_func,
				     bool display_sync_time) {
    char func_name_sync[strlen(func_name)+5];
    int n_func_indices, n_stack_indices;
    int *stack_indices = NULL, *func_indices = NULL;	
    int n_func_indices_sync, n_stack_indices_sync;
    int *func_indices_sync = NULL, *stack_indices_sync = NULL;

    if ((*display_func)->stack_id >= 0) {
       n_func_indices = 1;
       n_stack_indices = 1;
       func_indices = (int*) malloc (1 * sizeof(int));
       stack_indices = (int*) malloc (1 * sizeof(int));
       func_indices[0] = (*display_func)->func_id;
       stack_indices[0] = (*display_func)->stack_id;
    } else {
       vftr_find_function_in_table (func_name, &func_indices, &n_func_indices, true);
       vftr_find_function_in_stack (func_name, &stack_indices, &n_stack_indices, true);
    }
    (*display_func)->n_func_indices = n_func_indices;
    (*display_func)->func_indices = (int*)malloc (n_func_indices * sizeof(int));
    memcpy ((*display_func)->func_indices, func_indices, n_func_indices * sizeof(int));


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
        if (func_indices[i] < 0) break;
        // If synchronization times shall be displayed, only the inclusive time makes sense. For consistency,
        // we then show the inclusive time for all entries, not just that which have a synchronization barrier within them.
        // In the default case, we use the exclusive time.
        long long t1, t2, t1_sync, t2_sync;
        if (display_sync_time) {
            t1 = vftr_func_table[func_indices[i]]->prof_current.time_incl;
            t2 = vftr_func_table[func_indices[i]]->prof_previous.time_incl;
	    ///(*display_func)->this_mpi_time += vftr_func_table[func_indices[i]]->prof_current.time_incl;
            // Stack indices with n_func_indices_sync > 0 have a synchro entry.
	    ///if (n_func_indices_sync > 0) (*display_func)->this_sync_time += vftr_func_table[func_indices_sync[i]]->prof_current.time_incl;
	    if (n_func_indices_sync > 0) {
               t1_sync = vftr_func_table[func_indices_sync[i]]->prof_current.time_incl;
               t1_sync = vftr_func_table[func_indices_sync[i]]->prof_previous.time_incl;
            }
        } else {
            //printf ("Check time: %d %lld\n", func_indices[i], vftr_func_table[func_indices[i]]->prof_current.time_excl);
            t1 = vftr_func_table[func_indices[i]]->prof_current.time_excl;
            t2 = vftr_func_table[func_indices[i]]->prof_previous.time_excl;
	    ///(*display_func)->this_mpi_time += vftr_func_table[func_indices[i]]->prof_current.time_excl;
        }
        (*display_func)->this_mpi_time += t1 - t2;
        if (n_func_indices_sync > 0) (*display_func)->this_sync_time += t1_sync - t2_sync;
	(*display_func)->n_calls += vftr_func_table[func_indices[i]]->prof_current.calls;
	(*display_func)->mpi_tot_send_bytes += vftr_func_table[func_indices[i]]->prof_current.mpi_tot_send_bytes;
	(*display_func)->mpi_tot_recv_bytes += vftr_func_table[func_indices[i]]->prof_current.mpi_tot_recv_bytes;
	(*display_func)->properly_terminated &= !vftr_func_table[func_indices[i]]->open;
    }
    if (!(*display_func)->properly_terminated) return;
    long long all_times [vftr_mpisize], all_times_sync [vftr_mpisize];
#if defined(_MPI)
    PMPI_Allgather (&(*display_func)->this_mpi_time, 1, MPI_LONG_LONG_INT, all_times,
		 1, MPI_LONG_LONG_INT, MPI_COMM_WORLD);
    if (display_sync_time) {
	    PMPI_Allgather (&(*display_func)->this_sync_time, 1, MPI_LONG_LONG_INT, all_times_sync,
			 1, MPI_LONG_LONG_INT, MPI_COMM_WORLD);
    }
#else
    all_times[0] = (*display_func)->this_mpi_time;
#endif
    (*display_func)->t_max = 0;
    (*display_func)->t_sync_max = 0;
    (*display_func)->t_min = LLONG_MAX;
    (*display_func)->rank_min = -1;
    (*display_func)->t_sync_min = LLONG_MAX;
    (*display_func)->rank_max = -1;
    (*display_func)->t_avg = 0.0;
    (*display_func)->t_sync_avg = 0.0;
    (*display_func)->imbalance = 0.0;

    if ((*display_func)->n_calls == 0) return;

    long long sum_times = 0;
    long long sum_times_sync = 0;
    int n_count = 0;

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
#ifdef _MPI
       (*display_func)->imbalance = vftr_compute_mpi_imbalance (all_times, (*display_func)->t_avg);
#endif
       for (int i = 0; i < vftr_mpisize; i++) {	
       	  if (all_times[i] > 0) {
       		if (all_times[i] < (*display_func)->t_min) {
			(*display_func)->t_min = all_times[i];
			(*display_func)->rank_min = i;
			if (n_func_indices_sync > 0) (*display_func)->t_sync_min = all_times_sync[i];
		}
       		if (all_times[i] > (*display_func)->t_max) {
			(*display_func)->t_max = all_times[i];
			(*display_func)->rank_max = i;
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

display_function_t **vftr_create_display_functions (bool display_sync_time, int *n_display_funcs, bool use_all) {

   vftr_stackid_list_init();
   for (int i = 0; i < vftr_gStackscount; i++) {
       if (use_all || vftr_pattern_match(vftr_environment.print_stack_profile->value, vftr_gStackinfo[i].name)) {
          vftr_stackid_list_add (vftr_gStackinfo[i].locID, i);
       }
   }

   display_function_t **displ_f = (display_function_t**) malloc (vftr_n_print_stackids * sizeof(display_function_t*));

   int i_disp_f = 0;
   for (int i = 0; i < vftr_n_print_stackids; i++) {
       //int i_func = vftr_print_stackid_list[i];
       int i_stack = vftr_print_stackid_list[i].glob;
       int i_func = vftr_print_stackid_list[i].loc;
       if (!use_all) {
          bool name_already_there = false;
          for (int j = 0; j < i_disp_f; j++) {
             //if (!strcmp(displ_f[j]->func_name, vftr_gStackinfo[i_func].name)) {
             if (!strcmp(displ_f[j]->func_name, vftr_gStackinfo[i_stack].name)) {
               name_already_there = true;
               break;
             }
          }
          if (name_already_there) continue;
       }
       displ_f[i_disp_f] = (display_function_t*) malloc (sizeof(display_function_t));
       displ_f[i_disp_f]->func_name = strdup(vftr_gStackinfo[i_stack].name);
       vftr_is_traceable_mpi_function (displ_f[i_disp_f]->func_name, &(displ_f[i_disp_f]->is_mpi));
       displ_f[i_disp_f]->i_orig = i_disp_f;
       // Associating a stack ID with the display function only makes sense when we are in the complete_summary mode.
       // Otherwise all stacks with the same final function are added up. In that case, we set a negative dummy value.
       if (use_all) {
         displ_f[i_disp_f]->stack_id = i_stack;
         displ_f[i_disp_f]->func_id = i_func;

       } else {
         displ_f[i_disp_f]->stack_id = -1;
       }
       i_disp_f++;
   }
   *n_display_funcs = i_disp_f;

   for (int i = 0; i < *n_display_funcs; i++) {
      vftr_evaluate_display_function (displ_f[i]->func_name, &(displ_f[i]), display_sync_time);
   }

   qsort ((void*)displ_f, (size_t)(*n_display_funcs),
           sizeof (display_function_t *), vftr_compare_display_functions_tavg);
   return displ_f;
}

/**********************************************************************/

void vftr_print_function_statistics (FILE *fp_log, display_function_t **display_functions, int n_display_funcs, bool print_this_rank) {

    double total_time = 0;
    bool print_mpi_columns = false;
    if (!vftr_environment.all_mpi_summary->value) {
       for (int i = 0; i < n_display_funcs; i++) {
         if (display_functions[i]->properly_terminated) {
           total_time += display_functions[i]->this_mpi_time * 1e-6;
           print_mpi_columns |= display_functions[i]->is_mpi;
         }
       }
    }

    int table_width;
    if (print_this_rank) {
       fprintf (fp_log, "Total time spent in MPI for rank %d: %lf s\n", vftr_mpirank, total_time);
       fprintf (fp_log, "Imbalance computed as: max (T - T_avg)\n");

       //int n_columns = print_mpi_columns ? 10 : 7;
       // Per default, there are seven columns:
       //   - The function name
       //   - Number of calls
       //   - Average, mininum and maximum time
       //   - Imbalance
       //   - Time on this rank
       int n_columns = 7;
       // There are 3 more columns for MPI overviews:
       //   - Percentage of the total MPI time
       //   - Data size sent and received
       if (print_mpi_columns) {
          n_columns += 3;
       // If the complete overview is printed, add one column for global stack ID (or entire stack).
       } else if (vftr_environment.all_mpi_summary->value) {
          n_columns += 1;
       }
         
       column_t *columns = (column_t*) malloc (n_columns * sizeof(column_t));
       vftr_set_summary_column_formats (print_mpi_columns, n_display_funcs, display_functions, &columns, total_time);
   
       table_width = vftr_get_tablewidth_from_columns (columns, n_columns, true); 
       vftr_summary_print_header (fp_log, columns, table_width, print_mpi_columns);

       // Print all the display functions, but omit those without any calls.
       for (int i = 0; i < n_display_funcs; i++) {
          if (display_functions[i]->n_calls > 0 && display_functions[i]->properly_terminated) {
             vftr_summary_print_line (fp_log, display_functions[i], columns, total_time, print_mpi_columns);
          }
       }
    }

    //Print a final separator line.
    if (vftr_mpirank == 0) {
       for (int i = 0; i < table_width; i++) fprintf (fp_log, "-");
       fprintf (fp_log, "\n");
    }

    if (vftr_environment.print_stack_profile->value) {
	// Next, we print the function stack trees. But first, we need to undo the sorting done before. 
	// This is because inside of vftr_print_function_stack, the run times of individual stacks are
	// gathered across all ranks. If the order of the display functions is not the same for all ranks,
	// which can happen for example when one of them is an I/O rank, the gathered values can correspond
	// to different functions.  	
	//
	qsort ((void*)display_functions, (size_t)n_display_funcs,
	       sizeof (display_function_t *), vftr_compare_display_functions_iorig);

	if (vftr_environment.create_html->value) {
	   if (vftr_mpirank == 0) {
	      vftr_browse_print_index_html (display_functions, n_display_funcs);
	   }
        }

  	for (int i = 0; i < n_display_funcs; i++) {
                //if (!display_functions[i]->properly_terminated) continue;
		if (display_functions[i]->n_stack_indices == 0) {;
	 	   if (vftr_environment.create_html->value) {
		      vftr_browse_print_stacktree_page (NULL, true, display_functions, i,
		         				n_display_funcs, NULL, NULL, 0.0, 0, 0);
          	   }
		} else {
		   stack_leaf_t *stack_tree = NULL;
		   double *imbalances = (double*) malloc (vftr_func_table_size * sizeof (double));
		   vftr_stack_compute_imbalances (imbalances, display_functions[i]->n_stack_indices,
		   			       display_functions[i]->stack_indices);
                   if (print_this_rank) {
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
		         vftr_browse_print_stacktree_page (NULL, false, display_functions, i,
		           			        n_display_funcs, stack_tree->origin,
		           		                imbalances, (double)total_time * 1e-6, n_chars_max,
		           			        display_functions[i]->n_stack_indices);
		      }
                   }
		   free (stack_tree);
		   free (imbalances);
	       }
	}
  }

  free (display_functions);
}

/**********************************************************************/

void vftr_get_application_times_usec (long long time0, long long *total_runtime, long long *sampling_overhead_time,
			  	 long long  *mpi_overhead_time, long long *total_overhead_time, long long *application_time) {
   *total_runtime = time0 > 0 ? time0 : vftr_get_runtime_usec(); 
   *sampling_overhead_time = vftr_overhead_usec;
   *total_overhead_time = *sampling_overhead_time;
#if defined(_MPI)
   *mpi_overhead_time = vftr_mpi_overhead_usec;
   *total_overhead_time = *sampling_overhead_time + *mpi_overhead_time;
#else
   *mpi_overhead_time = 0;
#endif
   *application_time = *total_runtime - *total_overhead_time;
}

/**********************************************************************/

void vftr_print_profile_summary (FILE *fp_log, function_t **func_table, double total_runtime, double application_runtime,
				 double total_overhead_time, double sampling_overhead_time, double mpi_overhead_time) {

    //printf ("HUHU 1\n");
    fprintf(fp_log, "MPI size              %d\n", vftr_mpisize);
    fprintf(fp_log, "Total runtime:        %8.2f seconds\n", total_runtime);
    fprintf(fp_log, "Application time:     %8.2f seconds\n", application_runtime);
    fprintf(fp_log, "Overhead:             %8.2f seconds (%.2f%%)\n",
            total_overhead_time, 100.0*total_overhead_time/total_runtime);
    //printf ("HUHU 2\n");
#ifdef _MPI
    fprintf(fp_log, "   Sampling overhead: %8.2f seconds (%.2f%%)\n",
            sampling_overhead_time, 100.0*sampling_overhead_time/total_runtime);
    fprintf(fp_log, "   MPI overhead:      %8.2f seconds (%.2f%%)\n",
            mpi_overhead_time, 100.0*mpi_overhead_time/total_runtime);
#endif

    //printf ("HUHU 3: %d\n", vftr_stackscount);
    for (int i = 0; i < vftr_stackscount; i++) {
       profdata_t *prof_current  = &func_table[i]->prof_current;
       profdata_t *prof_previous = &func_table[i]->prof_previous;
       if (vftr_memtrace) {
          double mem = (double)(prof_current->event_count[vftr_n_hw_obs-1] - prof_previous->event_count[vftr_n_hw_obs-1]) / 1024 / 1024 / 1024;
          printf ("TEST: %s %d %lf %lf\n", func_table[i]->name, func_table[i]->id, mem, mem / (prof_current->calls - prof_previous->calls));
       }
    }
    //printf ("HUHU 4: %d\n", vftr_events_enabled);
    //fflush(stdout);
    vftr_scenario_expr_n_vars = 1;
    vftr_scenario_expr_n_formulas = 1;
    vftr_scenario_expr_counter_values = (double*)malloc (1 * sizeof(double));
    //vftr_events_enable = true;
    if (vftr_events_enabled) {
	unsigned long long total_cycles = 0;

        for (int i = 0; i < vftr_stackscount; i++) {
    	   if (func_table[i] == NULL) continue;
    	   if (func_table[i]->return_to && !func_table[i]->open && func_table[i]->prof_current.calls) {
              profdata_t *prof_current  = &func_table[i]->prof_current;
              profdata_t *prof_previous = &func_table[i]->prof_previous;
    	      total_cycles += prof_current->cycles - prof_previous->cycles;
              if (!prof_current->event_count || !prof_previous->event_count) continue;
    	      for (int j = 0; j < vftr_n_hw_obs; j++) {
    	          vftr_scenario_expr_counter_values[j] += (double)(prof_current->event_count[j] - prof_previous->event_count[j]);
    	      }
    	   }
        }

	//vftr_scenario_expr_evaluate_all (application_runtime, total_cycles);	
	//vftr_scenario_expr_print_summary (fp_log);

        fprintf (fp_log, "\nRaw counter totals\n"
            "------------------------------------------------------------\n"
            "%-37s : %20llu\n", 
            "Time Stamp Counter", total_cycles);
    	vftr_scenario_expr_print_raw_counters (fp_log);

    }
    fprintf (fp_log, "------------------------------------------------------------\n\n");
    fflush(fp_log);
}

/**********************************************************************/

void vftr_compute_line_content (function_t *this_func, int *n_calls, long long *t_excl, long long *t_incl, double *t_overhead) {
   profdata_t prof_current = this_func->prof_current;
   profdata_t prof_previous = this_func->prof_previous;
   *n_calls = (int)(prof_current.calls - prof_previous.calls);

   vftr_get_stack_times (prof_current, prof_previous, t_excl, t_incl);
   if (vftr_environment.show_overhead->value) {
      *t_overhead = this_func->overhead * 1e-6;
   } else {
      *t_overhead = 0.0;
   }

   if (vftr_events_enabled) {
   	vftr_fill_scenario_counter_values (vftr_scenario_expr_counter_values, vftr_scenario_expr_n_vars, 
   		prof_current, prof_previous);
       	//vftr_scenario_expr_evaluate_all (*t_excl * 1e-6, prof_current.cycles - prof_previous.cycles);
   }
  
   
}

/**********************************************************************/

void vftr_print_profile_line (FILE *fp_log, int local_stack_id, int global_stack_id,
			      long long runtime_usec, double sampling_overhead_time,
			      int n_calls, long long t_excl, long long t_incl, long long t_sum, double t_overhead,
			      char *func_name, char *caller_name, column_t *prof_columns) {
   int i_column = 0;
   vftr_prof_column_print (fp_log, prof_columns[i_column++], &n_calls, NULL, NULL);
   double t_part = (double)t_excl / (double)runtime_usec * 100;
   double t_cum = (double)t_sum / (double)runtime_usec * 100;
   vftr_print_stack_time (fp_log, n_calls, t_excl * 1e-6, t_incl * 1e-6, t_part, t_cum, &i_column, prof_columns);

   double t;
   if (vftr_environment.show_overhead->value) {
      vftr_prof_column_print (fp_log, prof_columns[i_column++], &t_overhead, NULL, NULL); 
      t = t_overhead / sampling_overhead_time * 100.0;
      vftr_prof_column_print (fp_log, prof_columns[i_column++], &t, NULL, NULL); 
      t = t_excl > 0 ? t_overhead / t_excl : 0.0;
      vftr_prof_column_print (fp_log, prof_columns[i_column++], &t, NULL, NULL); 
   }
   
   if (vftr_events_enabled) {
       	for (int i = 0; i < vftr_scenario_expr_n_formulas; i++) {
   	   //vftr_prof_column_print (fp_log, prof_columns[i_column++], &vftr_scenario_expr_formulas[i].value, NULL, NULL);
   	   vftr_prof_column_print (fp_log, prof_columns[i_column++], &vftr_scenario_expr_counter_values[i], NULL, NULL);
   	}
   }

   if (vftr_max_allocated_fields > 0) {
      double mem_max = (double)vftr_allocate_get_max_memory_for_stackid (local_stack_id);
      vftr_prof_column_print (fp_log, prof_columns[i_column++], &mem_max, NULL, NULL);
   }
   
   vftr_prof_column_print (fp_log, prof_columns[i_column++], func_name, NULL, NULL);
   if (caller_name) {
       vftr_prof_column_print (fp_log, prof_columns[i_column++], caller_name, NULL, NULL);
   } else {
       vftr_prof_column_print (fp_log, prof_columns[i_column++], "-/-", NULL, NULL);
   }
   
   vftr_prof_column_print (fp_log, prof_columns[i_column++], &global_stack_id, NULL, NULL);
   if (vftr_environment.show_stacks_in_profile->value) {
      vftr_prof_column_print (fp_log, prof_columns[i_column++], vftr_global_stack_strings[global_stack_id].s, NULL, NULL);
   }
   fprintf (fp_log, "\n");
}

/**********************************************************************/

void vftr_print_profile (FILE *fp_log, FILE *f_html, int *n_func_indices, long long time0,
                         int n_display_functions, display_function_t **display_functions) {
    int table_width;
    vftr_events_enabled = true;
    //vftr_scenario_expr_n_formulas = 1;

    if (!vftr_stackscount) return;

    function_t **func_table;

    for (int i = 0; i < vftr_scenario_expr_n_vars; i++) {
	vftr_scenario_expr_counter_values[i] = 0.0;
    }

    func_table = (function_t**) malloc (vftr_func_table_size * sizeof(function_t*));
    // Create a local copy of the global function table to sort it.
    memcpy (func_table, vftr_func_table, vftr_func_table_size * sizeof(function_t*));

    qsort ((void *)func_table, (size_t)vftr_stackscount, sizeof (function_t *), vftr_get_profile_compare_function());

    long long total_runtime_usec, sampling_overhead_time_usec,
	      total_overhead_time_usec, mpi_overhead_time_usec, application_runtime_usec;
    vftr_get_application_times_usec (time0, &total_runtime_usec, &sampling_overhead_time_usec, &mpi_overhead_time_usec, 
				     &total_overhead_time_usec, &application_runtime_usec);

    vftr_print_profile_summary (fp_log, func_table, total_runtime_usec * 1e-6,
				application_runtime_usec * 1e-6, total_overhead_time_usec * 1e-6,
				sampling_overhead_time_usec * 1e-6, mpi_overhead_time_usec * 1e-6);

    fprintf (fp_log, "Runtime profile");
    if (vftr_mpisize > 1) {
        fprintf (fp_log, " for rank %d", vftr_mpirank);
    }
    if (vftr_environment.prof_truncate->value) {
	fprintf (fp_log, ", truncated to %3.0f%%", vftr_environment.prof_truncate_cutoff->value);
    }
    fprintf (fp_log, ", %s", vftr_profile_sorting_method_string());
    fprintf (fp_log, " :\n\n");

    long long function_time = total_runtime_usec - sampling_overhead_time_usec;
    *n_func_indices = vftr_count_func_indices_up_to_truncate (func_table, function_time);
    int *func_indices = (int *)malloc (*n_func_indices * sizeof(int));
    vftr_fill_func_indices_up_to_truncate (func_table, function_time, func_indices);

    // Number of columns. Default: nCalls, exclusive & inclusive time, %abs, %cum,
    // function & caller name and stack ID (i.e. 8 columns). 
    int n_columns = 8;
    // Add one column for each hardware counter.
    //n_columns += vftr_scenario_expr_n_formulas;
    n_columns += 1;
    // If function overhead is displayed, add three more columns.
    if (vftr_environment.show_overhead->value) n_columns += 3;
    if (vftr_environment.show_stacks_in_profile->value) n_columns++;

    column_t *prof_columns = (column_t*) malloc (n_columns * sizeof(column_t));
    vftr_set_proftab_column_formats (func_table, function_time, sampling_overhead_time_usec * 1e-6,
				     *n_func_indices, func_indices, prof_columns);

    table_width = vftr_get_tablewidth_from_columns (prof_columns, n_columns, false);

    if (f_html != NULL) {
       vftr_browse_create_profile_header (f_html);
    }

    vftr_proftab_print_header (fp_log, prof_columns);
    vftr_print_dashes (fp_log, table_width);

    // All headers printed at this point
    // Next: the numbers

    long long cumulative_time = 0;
    for (int i = 0; i < *n_func_indices; i++) {
       int i_func = func_indices[i];
       if (func_table[i_func]->open) continue;
       int n_calls;
       long long t_excl, t_incl;
       double t_overhead;
       vftr_compute_line_content (func_table[i_func], &n_calls, &t_excl, &t_incl, &t_overhead);
       cumulative_time += t_excl;
       //printf ("Print line(%d): %s %d\n", vftr_mpirank, func_table[i_func]->name, func_table[i_func]->return_to == NULL);
       if (func_table[i_func]->return_to != NULL) {
          vftr_print_profile_line (fp_log, func_table[i_func]->id, func_table[i_func]->gid, 
   			        function_time, sampling_overhead_time_usec * 1e-6,
   			        n_calls, t_excl, t_incl, cumulative_time, t_overhead,
   				func_table[i_func]->name, func_table[i_func]->return_to->name, prof_columns);

       } else {
          //printf ("Init in profile line: %d %lld\n", i_func, t_excl);
          vftr_print_profile_line (fp_log, func_table[i_func]->id, func_table[i_func]->gid, 
   			        function_time, sampling_overhead_time_usec * 1e-6,
   			        n_calls, t_excl, t_incl, cumulative_time, t_overhead,
   				func_table[i_func]->name, NULL, prof_columns);
      }
      //printf ("Line printed: %d\n", vftr_mpirank);

       if (f_html != NULL) {
          bool mark_disp_f = false;
          for (int i_disp = 0; i_disp < n_display_functions; i_disp++) {
             if ((mark_disp_f = !strcmp(func_table[i_func]->name, display_functions[i_disp]->func_name))) break;
          }
          if (func_table[i_func]->return_to != NULL) {
	     vftr_browse_print_table_line (f_html, func_table[i_func]->gid,
	   			        function_time, sampling_overhead_time_usec * 1e-6,
	   				n_calls, t_excl, t_incl, cumulative_time, t_overhead,
	   			        func_table[i_func]->name, func_table[i_func]->return_to->name, prof_columns, mark_disp_f);
          } else {
	     vftr_browse_print_table_line (f_html, func_table[i_func]->gid,
	   			        function_time, sampling_overhead_time_usec * 1e-6,
	   				n_calls, t_excl, t_incl, cumulative_time, t_overhead,
	   			        func_table[i_func]->name, NULL, prof_columns, mark_disp_f);
          }
       }
    }

    if (f_html != NULL) vftr_browse_finalize_table(f_html);
    
    vftr_print_dashes (fp_log, table_width);
    fprintf (fp_log, "\n");
    fflush(fp_log);
    
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
	function_t *func1 = vftr_new_function (NULL, "init", NULL, false);
	function_t *func2 = vftr_new_function ((void*)addrs, "func2", func1, false);
	function_t *func3 = vftr_new_function ((void*)(addrs + 1), "func3", func1, false);	
	function_t *func4 = vftr_new_function ((void*)(addrs + 2), "func4", func3, false);
	function_t *func5 = vftr_new_function ((void*)(addrs + 3), "func5", func2, false);
	function_t *func6 = vftr_new_function ((void*)(addrs + 4), "func6", func2, false);
	function_t *func7 = vftr_new_function ((void*)(addrs + 5), "func4", func6, false);
	vftr_normalize_stacks();
	for (int i = 0; i < vftr_stackscount; i++) {
		vftr_func_table[i]->prof_current.calls = i + 1;
		vftr_func_table[i]->prof_current.cycles = 0;
		vftr_func_table[i]->prof_previous.cycles = 0;
		vftr_func_table[i]->prof_current.time_excl = (long long)(i+1) * 100000;
		vftr_func_table[i]->prof_previous.time_excl = (long long)(i+1) * 90000;
		vftr_func_table[i]->prof_current.time_incl =
			2 * vftr_func_table[i]->prof_current.time_excl;
		vftr_func_table[i]->prof_previous.time_incl =
			2 * vftr_func_table[i]->prof_previous.time_excl;
		vftr_test_runtime += vftr_func_table[i]->prof_current.time_excl
				   - vftr_func_table[i]->prof_previous.time_excl;
	}

	vftr_profile_wanted = true;
	vftr_mpisize = 1;
	vftr_overhead_usec = 0;
#ifdef _MPI
        vftr_mpi_overhead_usec = 0;
#endif
	vftr_print_profile (fp_out, NULL, &n, vftr_test_runtime, 0, NULL);
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

char *vftr_memory_unit_string (double value, int n_decimal_places) {
   char s[36], *s_unit;
   double v = value;
   vftr_memory_unit (&v, &s_unit);
   int n = vftr_count_digits_double (v) + n_decimal_places + 1;
   char f_tmp[16];
   sprintf (f_tmp, "%%%d.%df %%s", n, n_decimal_places);
   sprintf (s, f_tmp, v, s_unit);
   return strdup (s);
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

