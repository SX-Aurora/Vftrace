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

#include "vftr_scenarios.h"
#include "vftr_hwcounters.h"
#include "vftr_omp.h"
#include "vftr_environment.h"
#include "vftr_hooks.h"
#include "vftr_filewrite.h"
#include "vftr_functions.h"
#include "vftr_stacks.h"
#include "vftr_timer.h"
#include "vftr_setup.h"
#include "vftr_mpi_utils.h"

// File pointer of the log file
FILE *vftr_log;

// Individual vftrace-internal file id
char vftr_fileid[VFTR_FILEIDSIZE];

// Next sample time for each thread (one for each OpenMP thread)
long long *vftr_nextsampletime;

// The basename of Vftrace log files
char vftr_logfile_name[1024];

FILE **vftr_vfd_file;

// TODO: Explain
unsigned int vftr_admin_offset;
unsigned int vftr_samples_offset;

void vftr_init_vfd_file (char *basename, int task_digits, int thread_digits) {
	char *trace_file_name_format = malloc (1024 * sizeof(char));
 	/* "dir/ident_%0<taskDigits>d_%0<threadDigits>d.vfd" */
        sprintf (trace_file_name_format, "%s/%s_%%0%dd_%%0%dd.vfd", 
             vftr_environment->output_directory->value,
	     basename, task_digits, thread_digits);

	for (int omp_thread  = 0; omp_thread < vftr_omp_threads; omp_thread++) {
	    size_t size = vftr_environment->bufsize->value * 1024 * 1024;
	    char filename[1024];
	    sprintf (filename, trace_file_name_format, vftr_mpirank, omp_thread);
	    FILE *fp = fopen (filename, "w+");
	    assert (fp);
	    char *buf = (char *) malloc( size );
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
	    fwrite (&vftr_omp_threads, sizeof(int), 1, fp ); 
	    fwrite (&omp_thread, sizeof(int), 1, fp ); 
	
	    if (omp_thread == 0) vftr_admin_offset = (unsigned int) ftell (fp);
	
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
	    fwrite (zeroint, sizeof(unsigned int), 5, fp);
	    // Store global information about hardware scenarios
	    vftr_write_scenario_header_to_vfd (fp);	

	    if (omp_thread == 0) vftr_samples_offset = (unsigned int) ftell (fp);
	    vftr_vfd_file[omp_thread] = fp;
	}
}

/**********************************************************************/

void vftr_finalize_vfd_file (long long finalize_time, int signal_number) {
    for (int omp_thread = 0; omp_thread < vftr_omp_threads; omp_thread++) {

        if (vftr_env_do_sampling () && signal_number != SIGUSR1) {

            unsigned int stackstable_offset = (unsigned int) ftell (vftr_vfd_file[omp_thread]);
            vftr_write_stacks (vftr_vfd_file[omp_thread], 0, vftr_froots[0]);

	    // It is unused ?
            unsigned int profile_offset = 0;

            double runtime = finalize_time * 1.0e-6;
            double zerodouble[] = { 0., 0. };

            // Update trace info in header and close
            fseek (vftr_vfd_file[omp_thread], vftr_admin_offset, SEEK_SET);
            fwrite(&vftr_mpisize, sizeof(int), 1, vftr_vfd_file[omp_thread]); 
            fwrite(&vftr_mpirank, sizeof(int),1, vftr_vfd_file[omp_thread]); 
            fwrite(&zerodouble, sizeof(double),	1, vftr_vfd_file[omp_thread]); 
	    fwrite(&vftr_inittime, sizeof(long long), 1, vftr_vfd_file[omp_thread]);
            fwrite(&runtime, sizeof(double), 1, vftr_vfd_file[omp_thread]);
            fwrite(&vftr_samplecount[omp_thread], sizeof(unsigned int), 1, vftr_vfd_file[omp_thread]);
            fwrite(&vftr_stackscount, sizeof(unsigned int), 1, vftr_vfd_file[omp_thread]);
            fwrite(&stackstable_offset, sizeof(unsigned int), 1, vftr_vfd_file[omp_thread]);
            fwrite(&vftr_samples_offset, sizeof(unsigned int), 1, vftr_vfd_file[omp_thread]);
	    fwrite(&profile_offset, sizeof(unsigned int), 1, vftr_vfd_file[omp_thread]);
            fclose (vftr_vfd_file[omp_thread]);

        }
    }

}

/**********************************************************************/

void vftr_write_to_vfd(long long runtime, unsigned long long cycles, int stack_id, unsigned int sid, int me) {
    fwrite (&sid, sizeof(unsigned int), 1, vftr_vfd_file[me]);
    fwrite (&stack_id, sizeof(int), 1, vftr_vfd_file[me]);
    fwrite (&runtime, sizeof(long long), 1, vftr_vfd_file[me]);

    vftr_write_observables_to_vfd (cycles, vftr_vfd_file[me]);

    vftr_nextsampletime[me] = runtime + vftr_interval;
    vftr_prevsampletime [me] = runtime;
    vftr_samplecount[me]++;
}

/**********************************************************************/

#ifdef _MPI
// Store the message information in a vfd file
void vftr_store_message_info(vftr_direction dir, int count, int type_idx,
                             int type_size, int rank, int tag,
                             long long tstart, long long tend) {
   
   int omp_thread = 0;
   int sid = SID_MESSAGE;
   fwrite(&sid, sizeof(int), 1, vftr_vfd_file[omp_thread]);
   fwrite(&dir, sizeof(int), 1, vftr_vfd_file[omp_thread]);
   fwrite(&rank, sizeof(int), 1, vftr_vfd_file[omp_thread]);
   fwrite(&type_idx, sizeof(int), 1, vftr_vfd_file[omp_thread]);
   fwrite(&count, sizeof(int), 1, vftr_vfd_file[omp_thread]);
   fwrite(&type_size, sizeof(int), 1, vftr_vfd_file[omp_thread]);
   fwrite(&tag, sizeof(int), 1, vftr_vfd_file[omp_thread]);
   fwrite(&tstart, sizeof(long long), 1, vftr_vfd_file[omp_thread]);
   fwrite(&tend, sizeof(long long), 1, vftr_vfd_file[omp_thread]);

   vftr_samplecount[omp_thread]++;
}
#endif

/**********************************************************************/

void vftr_write_profile () {
    int            i, j, tid, zero = 0;
    double         rtime;
    unsigned long long      totalCycles, calls, cycles, *ec;
    evtcounter_t    *evc;
    FILE           *fp = vftr_vfd_file[0];

    function_t   **funcTable;

    if (!vftr_stackscount)return;
    funcTable = vftr_func_table;

    ec = (long long *) malloc (vftr_n_hw_obs * sizeof(long long));
    for (j = 0; j < vftr_n_hw_obs; j++) {
	ec[j] = 0;
    }

    totalCycles = 0;
 
    /* Sum all cycles and counts */
    for (i = 0; i < vftr_stackscount; i++ ) {
	if (funcTable[i] && funcTable[i]->ret && funcTable[i]->prof_current[0].calls) {
            profdata_t *prof_current = &funcTable[i]->prof_current[0];
	    totalCycles += prof_current->cycles;
            if (!prof_current->event_count) continue;
            for (j = 0; j < vftr_n_hw_obs; j++) {
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

    for( i=0; i<vftr_stackscount; i++ ) {
        if( !funcTable[i] ) continue;
        for( tid=0; tid<vftr_omp_threads; tid++ ) {
            profdata_t *prof_current  = &funcTable[i]->prof_current [tid];
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
		profdata_t *prof_current = &funcTable[i]->prof_current[0];
		profdata_t *prof_previous = &funcTable[i]->prof_previous[0];
		/* If function has a caller and has been called */
		if (!(funcTable[i]->ret && prof_current->calls)) continue;
		indices[j++] = i;
		get_stack_times (prof_current, prof_previous, runtime, &t_excl, &t_incl, &t_part);
		ctime += t_part;
		if (vftr_environment->prof_truncate->value && ctime > max_ctime) break;
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
		profdata_t *prof_current = &funcTable[i]->prof_current[0];
		profdata_t *prof_previous = &funcTable[i]->prof_previous[0];
		/* If function has a caller and has been called */
		if (!(funcTable[i]->ret && prof_current->calls)) continue;
		
		n_indices++;

		get_stack_times (prof_current, prof_previous, runtime, &t_excl, &t_incl, &t_part);
		ctime += t_part;
		if (vftr_environment->prof_truncate->value && ctime > max_ctime) break;
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
	int fidp;
	for (format->fid = 0, ev = vftr_gStackscount; ev; ev /= 10, format->fid++);
	for (format->rank = 0, ev = vftr_mpisize; ev; ev /= 10, format->rank++);
	for (format->thread = 0, ev = vftr_omp_threads; ev; ev /= 10, format->thread++);
	//if (format->fid < 4) format->fid = 2;
	//if (format->rank < 2) format->rank = 2;
	//if (format->thread < 2) format->thread = 2;
	format->fid = 2;
	format->rank = 2;
	format->thread = 2;
	format->n_calls = MIN_CALLS_NCHAR;
	format->func_name = MIN_FUNC_NCHAR;
	format->caller_name = MIN_CALLER_NCHAR;
	format->incl_time = MIN_INCTIME_NCHAR;
        format->excl_time = MIN_EXCLTIME_NCHAR;
	// 
	for (int i = 0; i < n_indices; i++) {
		int i_func = indices[i];
		profdata_t *prof_current = &funcTable[i_func]->prof_current[0];
		profdata_t *prof_previous = &funcTable[i_func]->prof_previous[0];

		if (vftr_events_enabled) {
			fill_scenario_counter_values (scenario_expr_counter_values,
				scenario_expr_n_vars, prof_current, prof_previous);
		}

        	int k = strlen(funcTable[i_func]->name);
		if (k > format->func_name) format->func_name = k;
		function_t *func;
        	if (func = funcTable[i_func]->ret) {
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
		    //int cycles = prof_current->cycles - prof_previous->cycles;
		    unsigned long long cycles = prof_current->cycles - prof_previous->cycles;
		    scenario_expr_evaluate_all (t_excl, cycles);
		    scenario_expr_set_formats ();
	        }
	}
	if (format->excl_time < 5) format->excl_time = 5;
    	if (format->incl_time < 5) format->incl_time = 5;
}

/**********************************************************************/

void vftr_print_profile (FILE *pout, int *ntop, long long time0) {
    float          pscale, ctime;
    double         rtime, tohead, pohead, tend, tend2;
    double         clockFreq;
    unsigned long long      totalCycles, calls, cycles;
    evtcounter_t    *evc0, *evc1, *evc;
    
    int            n, k, fid;
    int            linesize = 64; /* L3 cache line size */

    int            offset, tableWidth;

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

    funcTable = vftr_func_table;

    qsort( (void *)funcTable, (size_t)vftr_stackscount, sizeof( function_t *),
	   vftr_compare );    

    if (!vftr_profile_wanted)  return;

    totalCycles = 0;
    ctime = 0;
 
    /* Sum all cycles and counts */
    for (int i = 0; i < vftr_stackscount; i++) {
	if (funcTable[i] == NULL) continue;
	if (funcTable[i]->ret && funcTable[i]->prof_current[0].calls) {
            profdata_t *prof_current  = &funcTable[i]->prof_current[0];
            profdata_t *prof_previous = &funcTable[i]->prof_previous[0];
	    totalCycles += prof_current->cycles - prof_previous->cycles;
            if (!prof_current->event_count || !prof_previous->event_count) continue;
	    for (int j = 0; j < scenario_expr_n_vars; j++) {
		scenario_expr_counter_values[j] += (double)(prof_current->event_count[j] - prof_previous->event_count[j]);
	    }
	}
    }
    // identify the thread id
    int thread_id = OMP_GET_THREAD_NUM;
    double total_runtime = vftr_get_runtime_usec() * 1.0e-6;
    double overhead_time = vftr_overhead_usec[thread_id] * 1.0e-6;
    double application_runtime = total_runtime - overhead_time;
    rtime = application_runtime;

    /* Print profile info */

    fprintf(pout, "MPI size          %d\n", vftr_mpisize);
    fprintf(pout, "Total runtime:    %8.2f seconds\n", total_runtime);
    fprintf(pout, "Application time: %8.2f seconds\n", application_runtime);
    fprintf(pout, "Overhead:         %8.2f seconds (%.2f%%)\n", overhead_time,
            100.0*overhead_time/total_runtime);

    /* Print overall info */
    if (vftr_events_enabled) {
	scenario_expr_evaluate_all (rtime, totalCycles);	
	scenario_expr_print_summary (pout);
    }

    /* Print all raw counter totals */
    fprintf( pout, "\nRaw counter totals\n"
            "------------------------------------------------------------\n"
            "%-37s : %20lld\n", 
            "Time Stamp Counter", totalCycles  );
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
        if (vftr_omp_threads>1) {
		fprintf (pout, " for rank %d, thread 0", vftr_mpirank);
        } else {
                 fprintf( pout, " for rank %d",           vftr_mpirank );
	}
    } else {
        if (vftr_omp_threads > 1) fprintf( pout, " for thread 0" );
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
        profdata_t *prof_current   = &funcTable[i_func]->prof_current [0];
        profdata_t *prof_previous  = &funcTable[i_func]->prof_previous[0];

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

	if (funcTable[i_func]->ret) {
            fprintf (pout, "%s", funcTable[i_func]->ret->name);
            for (int j = strlen(funcTable[i_func]->ret->name); j <= formats->caller_name; j++) {
                fputc (' ', pout);
            }
        }


        bool multiTask = vftr_mpisize > 1;
	fid = (pout != stdout && multiTask) ? funcTable[i_func]->gid : funcTable[i_func]->id;
        fprintf (pout, fmtfid, fid);
        fprintf (pout, "\n");

    }
    
    output_dashes_nextline (tableWidth, pout);   
    fprintf( pout, "\n" );
}
