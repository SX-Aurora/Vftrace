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

#include <fcntl.h>
#include <limits.h>
#include <string.h>
#include <sys/stat.h>
#include <assert.h>
#include <unistd.h>

#include "vftr_symbols.h"
#include "vftr_scenarios.h"
#include "vftr_hwcounters.h"
#include "vftr_version.h"
#include "vftr_environment.h"
#include "vftr_setup.h"
#include "vftr_dlopen.h"
#include "vftr_timer.h"
#include "vftr_fileutils.h"
#include "vftr_filewrite.h"
#include "vftr_browse.h"
#include "vftr_signals.h"
#include "vftr_stacks.h"
#include "vftr_hooks.h"
#include "vftr_timer.h"
#include "vftr_functions.h"
#include "vftr_mallinfo.h"
#include "vftr_allocate.h"
#include "vftr_cuda.h"

bool vftr_timer_end;

int vftr_mpirank;
int vftr_mpisize;
unsigned int vftr_function_samplecount;
unsigned int vftr_message_samplecount;

bool vftr_do_stack_normalization;

char *vftr_start_date;
char *vftr_end_date;

bool in_vftr_finalize;

void vftr_print_disclaimer_full (FILE *fp) {
    fprintf (fp, 
        "\nThis program is free software; you can redistribute it and/or modify\n"
        "it under the terms of the GNU General Public License as published by\n"
        "the Free Software Foundation; either version 2 of the License , or\n"
        "(at your option) any later version.\n\n"
        "This program is distributed in the hope that it will be useful,\n"
        "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
        "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
        "GNU General Public License for more details.\n\n"
        "You should have received a copy of the GNU General Public License\n"
        "along with this program. If not, write to\n\n"
        "   The Free Software Foundation, Inc.\n"
        "   51 Franklin Street, Fifth Floor\n"
        "   Boston, MA 02110-1301  USA\n\n" );
}

/**********************************************************************/

void vftr_print_disclaimer (FILE *fp, bool no_date) {
    int v_major = MAJOR_VERSION;
    int v_minor = MINOR_VERSION;
    int rev = REVISION;
    fprintf (fp, "Vftrace version %d.%d.%d\n", v_major, v_minor, rev);
    fprintf (fp, "Runtime profile for application: %s\n", "");
    if (!no_date) fprintf (fp, "Start Date: %s\n", vftr_start_date); 
    fprintf (fp, 
        "This is free software with ABSOLUTELY NO WARRANTY.\n"
        "For details: use vftrace with environment variable VFTR_LICENSE=yes\n");
}

void vftr_print_startup_message(FILE *fp) {
#define STRINGIFY_(x...) #x
#define STRINGIFY(x) STRINGIFY_(x)
   char *versionstr = STRINGIFY(_VERSION);
   char *bugreportstr = STRINGIFY(_BUGREPORT);
#undef STRINGIFY
#undef STRINGIFY_

   if (vftr_rank_needs_logfile()) {
      fprintf(fp, "This program is traced by vftrace %s\n", versionstr);
      fprintf(fp, "Please report bugs to \n   %s\n", bugreportstr);
   }
}
/**********************************************************************/

void vftr_get_mpi_info (int *rank, int *size) {
#ifdef _MPI
// At this point, MPI_Init has not been called yet, so we cannot
// use MPI_Comm_size or MPI_Comm_rank. Instead, we have to rely
// on these environment variables set by various MPI implementations.
    if (getenv ("PMI_RANK")) {
        *rank = atoi (getenv("PMI_RANK"));
        *size = atoi (getenv("PMI_SIZE"));
    } else if (getenv ("OMPI_COMM_WORLD_RANK")) {
        *rank = atoi (getenv("OMPI_COMM_WORLD_RANK"));
	*size = atoi (getenv("OMPI_COMM_WORLD_SIZE"));
    } else if (getenv ("PMI_ID")) {
        *rank = atoi (getenv("PMI_ID"));
	*size = atoi (getenv("MPIRUN_NPROCS"));
    } else if (getenv ("MPIRUN_RANK")) {
        *rank = atoi (getenv ("MPIRUN_RANK"));
	*size = atoi (getenv ("MPIRUN_NPROCS"));
    } else if (getenv ("MPIRANK")) {
        *rank = atoi (getenv ("MPIRANK"));
        /* MPISIZE not set by MPI/SX, will be set after mpi_init call */
	*size = atoi (getenv ("MPISIZE"));
    } else if (getenv ("SLURM_PROCID")) {
	*rank = atoi (getenv ("SLURM_PROCID"));
	char *s;
	if (s = getenv ("SLURM_NPROCS")) *size = atoi (s);
    } else {
	// Cannot find out how many MPI ranks there are, assume only one.
	*rank = 0;
	*size = 1;
    }
#else
    // No MPI, only one rank exists.
    *rank = 0;
    *size = 1;
#endif
}
	
/**********************************************************************/

void vftr_set_start_date () {
   time_t t;
   time(&t);
   vftr_start_date = strdup(ctime (&t));
}

void vftr_set_end_date () {
   time_t t;
   time(&t);
   vftr_end_date = strdup(ctime (&t));
}

/**********************************************************************/

void vftr_initialize() {
    in_vftr_finalize = false;
    // set the timer reference point for this process
    vftr_set_local_ref_time();
    vftr_set_start_date();
    
    // measure overhead induced by vftrace
    long long overhead_time_start = vftr_get_runtime_usec();

    vftr_read_environment ();
    if (vftr_off()) {
	return;
    }
    atexit (vftr_finalize);
    vftr_get_mpi_info (&vftr_mpirank, &vftr_mpisize);
    vftr_set_logfile_ranks();
    vftr_set_mpi_summary_ranks();
    vftr_assert_environment ();
    setup_vftr_cuda();

    if (vftr_environment.show_startup->value) {
       vftr_print_startup_message(stdout);
    }

    vftr_do_stack_normalization = !vftr_environment.no_stack_normalization->value;
    vftr_setup_signals();
	
    lib_opened = 0;
    vftr_timelimit = LONG_MAX;

    // No buffering for messages going directly to stdout
    setvbuf (stdout, NULL, _IOLBF, (size_t)0);

    sprintf (vftr_fileid, "VFTRACE %07d", 
	(100 * MAJOR_VERSION + MINOR_VERSION) * 100 + REVISION);

    vftr_overhead_usec = 0ll;
    vftr_prog_cycles = 0ll;

    vftr_program_path = vftr_get_program_path ();
    vftr_logfile_name = vftr_create_logfile_name (vftr_mpirank, vftr_mpisize, "log");

    vftr_log = fopen (vftr_logfile_name, "w+");
    assert (vftr_log);
    // Do not buffer when writing into the log file
    setvbuf (vftr_log, NULL, _IOLBF, (size_t)0);

    if (vftr_rank_needs_logfile()) {
       if (vftr_environment.license_verbose->value) {
	  vftr_print_disclaimer_full (vftr_log);
       } else {
	  vftr_print_disclaimer (vftr_log, false);
       }
    }
        
    if (vftr_rank_needs_logfile()) {
       vftr_check_env_names (vftr_log);
    }


    if (vftr_create_symbol_table (vftr_mpirank)) {
	// No symbol table has been found. Switch of symbol table.
	printf ("Vftrace could not find any parseable symbol tables associated with your executable\n");
	printf ("It will not be active for this application run\n");
	printf ("Consider recompiling Vftrace using -D__VMAP_OFFSET\n");
	vftr_switch_off();	
    }

    if (vftr_off()) {
        vftr_events_enabled = false;
    }

    memset (&vftr_prof_data, 0, sizeof(profdata_t));

    vftr_init_mallinfo();

    // initialize the stack variables and tables

    vftr_nextsampletime = 0ll;
    vftr_prevsampletime = 0;

    /* Init event counters */
    vftr_n_hw_obs = 0;
    if (vftr_environment.scenario_file->set) {
	vftr_events_enabled = !vftr_init_hwc (vftr_environment.scenario_file->value);
    } else {
	vftr_events_enabled = false;
    }
    if (vftr_n_hw_obs < 0) {
        fprintf(vftr_log, "error initializing H/W counters\n");
        vftr_events_enabled = false;
    }

    if (vftr_memtrace) {
       vftr_init_hwc_memtrace();
    }

    vftr_initialize_stacks();

    if (vftr_n_hw_obs > 0) {
       vftr_prof_data.events[0] = (long long *) malloc (vftr_n_hw_obs * sizeof(long long));
       vftr_prof_data.events[1] = (long long *) malloc (vftr_n_hw_obs * sizeof(long long));
       memset (vftr_prof_data.events[0], 0, vftr_n_hw_obs * sizeof(long long));
       memset (vftr_prof_data.events[1], 0, vftr_n_hw_obs * sizeof(long long));
    }

    vftr_initcycles = vftr_get_cycles();
    // convert the sampletime and timelimit to microseconds
    vftr_interval  = (long long) (vftr_environment.sampletime->value * 1.0e6);
    assert (vftr_interval > 0ll);
    vftr_timelimit = vftr_environment.stoptime->value * 1000000ll;

    vftr_sorttime = 15.;
    vftr_sorttime_growth = 2.;

    /* Create VFD files */
    if (vftr_env_do_sampling ()) {
	vftr_init_vfd_file ();
    }
    
    vftr_profile_wanted = vftr_rank_needs_logfile();

    if (vftr_environment.print_stacks_for->set) {
        char *vftr_print_groups = vftr_environment.print_stacks_for->value;
        while (*vftr_print_groups && !vftr_profile_wanted) {
            int group_base, group_size;
            char *p;
	    // Loop to the end of the group, indicated by ","
            for (p = vftr_print_groups; *p && *p != ','; p++);
	    // sscanf returns the number of variables which have been filled
            switch (sscanf (vftr_print_groups, "%d:%d", &group_base, &group_size))  {
            case 1:
	      // Only one rank is to be printed. Check if it is the own rank.
              vftr_profile_wanted = vftr_mpirank == group_base;
	      break;
            case 2:
	      // Check if the own rank is in the interval given by [group_base, group_base + group_size).
              vftr_profile_wanted = vftr_mpirank >= group_base &&
                                    vftr_mpirank <  group_base + group_size;
            }
	    // Check if there is anything behind the comma of the environment variable. If so, proceed.
            vftr_print_groups = *p ? p + 1 : p;
        }
    }

    fflush (stdout);
    vftr_initcycles = vftr_get_cycles();
    
    // get the time to estimate vftrace overhead
    long long overhead_time_end = vftr_get_runtime_usec();
    vftr_overhead_usec += overhead_time_end - overhead_time_start;
}

/**********************************************************************/

void vftr_finalize() {
    in_vftr_finalize = true;

    if (vftr_off())  return;
    vftr_set_end_date();

    long long finalize_time = vftr_get_runtime_usec();

    final_vftr_cuda();

    vftr_timer_end = true;

#ifdef _MPI
    // check whether an MPI-init was actually called
    int was_mpi_initialized;
    PMPI_Initialized(&was_mpi_initialized);
#endif

    // Mark end of non-parallel interval
    if (vftr_env_do_sampling()) {
        vftr_write_to_vfd (finalize_time, NULL, 0, SID_EXIT);
    }

    if (vftr_environment.strip_module_names->value) {
	vftr_strip_all_module_names ();
    }

    if (vftr_environment.demangle_cpp->value) {
#ifdef _LIBERTY_AVAIL
       vftr_demangle_all_func_names();
#endif
    }
    
    FILE *f_html = NULL;
    display_function_t **display_functions = NULL;
    int n_display_functions = 0;
#ifdef _MPI
    if (vftr_do_stack_normalization && was_mpi_initialized) {
#else
    if (vftr_do_stack_normalization) {
#endif
       vftr_normalize_stacks();

       if (vftr_env_need_display_functions()) {
          display_functions = vftr_create_display_functions (vftr_environment.mpi_show_sync_time->value,
                                                             &n_display_functions,
                                                             strcmp(vftr_environment.mpi_summary_for_ranks->value, "")); 
       }

       if (vftr_environment.create_html->value) {
          vftr_browse_create_directory ();
          f_html = vftr_browse_init_profile_table (display_functions, n_display_functions);
       }
    }

    function_t **sorted_func_table = vftr_get_sorted_func_table ();
    vftr_prof_times_t prof_times = vftr_get_application_times_all (vftr_get_runtime_usec());
    int n_functions_top = vftr_count_func_indices_up_to_truncate (sorted_func_table,
                      prof_times.t_usec[TOTAL_TIME] - prof_times.t_usec[SAMPLING_OVERHEAD]);

    if (vftr_profile_wanted) {
       if (vftr_do_stack_normalization) vftr_create_global_stack_strings ();
       vftr_print_profile (vftr_log, sorted_func_table, n_functions_top, prof_times, n_display_functions, display_functions);
    }
#ifdef _MPI
    if (was_mpi_initialized) {
       if (vftr_do_stack_normalization && (vftr_environment.print_stack_profile->value || vftr_needs_mpi_summary())) {
          // Inside of vftr_print_function_statistics, we use an MPI_Allgather to compute MPI imbalances. Therefore,
          // we need to call this function for every rank, but give it the information of vftr_profile_wanted
          // to avoid unrequired output.
          vftr_print_function_statistics (vftr_log, display_functions, n_display_functions, vftr_profile_wanted);
          if (vftr_environment.create_html->value) vftr_print_function_statistics_html (display_functions, prof_times, n_display_functions, vftr_profile_wanted);
       }
    }
#endif

    if (f_html != NULL) {
       vftr_print_html_profile(f_html, sorted_func_table, n_functions_top, prof_times, n_display_functions, display_functions, vftr_get_runtime_usec());
    }

    vftr_print_gpu_summary (vftr_log);
 
    if (vftr_profile_wanted && vftr_do_stack_normalization) {
        vftr_print_global_stacklist(vftr_log);
    }

    if (vftr_env_do_sampling()) vftr_finalize_vfd_file (finalize_time);
    if (vftr_events_enabled && vftr_stop_hwc() < 0) {
	fprintf(vftr_log, "error stopping H/W counters, ignored\n");
    }

    if (vftr_memtrace) vftr_finalize_mallinfo();
    if (vftr_max_allocated_fields > 0) vftr_allocate_finalize(vftr_log);

    if (vftr_environment.print_env->value) vftr_print_environment(vftr_log);

    vftr_show_user_traced_stacktrees (vftr_log);
    
    if (vftr_log) {
    	bool is_empty = (ftello (vftr_log) == (off_t)0);
    	if (is_empty) unlink (vftr_logfile_name);

    	fclose (vftr_log);
    }

    free (sorted_func_table);
    free (display_functions);
    vftr_switch_off();
    in_vftr_finalize = false;
}

/**********************************************************************/

// vftr_finalize has to be called in the wrapper of MPI_Finalize, both for C and Fortran.
// This is the corresponding symbol for Fortran, with an added "_".
// It always calls vftr_finalize with active stack normalization ("true" argument), since
// this is the standard way to terminate.
void vftr_finalize_() {
	vftr_finalize();
#ifdef _MPI
	PMPI_Barrier (MPI_COMM_WORLD);
#endif
}

/**********************************************************************/
