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
#include "vftr_allocate.h"

bool vftr_timer_end;

int vftr_mpirank;
int vftr_mpisize;
unsigned int vftr_function_samplecount;
unsigned int vftr_message_samplecount;

char *vftr_start_date;
char *vftr_end_date;

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

void vftr_print_disclaimer (FILE *fp) {
    int v_major = MAJOR_VERSION;
    int v_minor = MINOR_VERSION;
    int rev = REVISION;
    fprintf (fp, "Vftrace version %d.%d.%d\n", v_major, v_minor, rev);
    fprintf (fp, "Runtime profile for application: %s\n", "");
    fprintf (fp, "Start Date: %s\n", vftr_start_date); 
    fprintf (fp, 
        "This is free software with ABSOLUTELY NO WARRANTY.\n"
        "For details: use vftrace with environment variable VFTR_LICENSE\n"
        "set to 1, or run \"vfview -w\", or consult the COPYRIGHT file.\n" );
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
    // set the timer reference point for this process
    vftr_set_local_ref_time();
    vftr_set_start_date();
    
    // measure overhead induced by vftrace
    long long overhead_time_start = vftr_get_runtime_usec();

    vftr_read_environment ();
    if (vftr_off()) {
	return;
    }
    vftr_assert_environment ();
	
    lib_opened = 0;

    vftr_timelimit = LONG_MAX;

    // No buffering for messages going directly to stdout
    setvbuf (stdout, NULL, _IOLBF, (size_t)0);

    sprintf (vftr_fileid, "VFTRACE %07d", 
	(100 * MAJOR_VERSION + MINOR_VERSION) * 100 + REVISION);

    vftr_overhead_usec = 0ll;
  
    vftr_prog_cycles = 0ll;

    vftr_program_path = vftr_get_program_path ();
    vftr_get_mpi_info (&vftr_mpirank, &vftr_mpisize);

    vftr_logfile_name = vftr_create_logfile_name (vftr_mpirank, vftr_mpisize, "log");

    vftr_log = fopen (vftr_logfile_name, "w+");
    assert (vftr_log);
    // Do not buffer when writing into the log file
    setvbuf (vftr_log, NULL, _IOLBF, (size_t)0);

    if (vftr_mpirank == 0) {
       if (vftr_environment.license_verbose->value) {
	  vftr_print_disclaimer_full (vftr_log);
       } else {
	  vftr_print_disclaimer (vftr_log);
       }
    }

    if (vftr_create_symbol_table (vftr_mpirank, NULL)) {
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

    // initialize the stack variables and tables
    vftr_initialize_stacks();

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

    if (vftr_n_hw_obs  > 0) {
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
    
    vftr_profile_wanted = (vftr_environment.logfile_all_ranks->value) || (vftr_mpirank == 0);

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

    /* Define signal handlers */
    if (!vftr_environment.signals_off->value) {
	vftr_define_signal_handlers ();
    }

    fflush (stdout);
    vftr_initcycles = vftr_get_cycles();
    
    // get the time to estimate vftrace overhead
    long long overhead_time_end = vftr_get_runtime_usec();
    vftr_overhead_usec += overhead_time_end - overhead_time_start;
}

/**********************************************************************/

void vftr_calc_tree_format (function_t *func) {
    long long   fcalls, ftime;

    if (func == NULL) return;

    fcalls = func->prof_current.calls;
    ftime  = func->prof_current.cycles;
    if (vftr_maxtime < ftime) vftr_maxtime = ftime;

    int n = func->levels;

    /* Recursive search of callees */
    int i;
    function_t *f;
    for (i = 0, f = func->first_in_level; i < n; i++,f = f->next_in_level) {
        vftr_calc_tree_format (f);
    }
}

/**********************************************************************/

void vftr_finalize() {
    int ntop = 0;
    function_t **funcTable;

    if (vftr_off())  return;
    vftr_set_end_date();

    // get the total runtime
    long long finalize_time = vftr_get_runtime_usec();

    vftr_timer_end = true;

    // Mark end of non-parallel interval
    if (vftr_env_do_sampling()) {
        vftr_write_to_vfd (finalize_time, NULL, NULL, 0, SID_EXIT);
    }

    if (vftr_environment.strip_module_names->value) {
	vftr_strip_all_module_names ();
    }
    
    vftr_normalize_stacks();
    vftr_calc_tree_format (vftr_froots);

    display_function_t **display_functions;
    int n_display_functions;
    //if (vftr_environment.print_stack_profile->value || vftr_environment.create_html->value) {
       display_functions = vftr_create_display_functions (vftr_environment.mpi_show_sync_time->value, &n_display_functions, true); 
    //}

    FILE *f_html = NULL;
    if (vftr_environment.create_html->value) {
       vftr_browse_create_directory ();
       f_html = vftr_browse_init_profile_table (display_functions, n_display_functions);
    }

    //vftr_print_profile (vftr_log, display_functions, n_display_functions, &ntop, vftr_get_runtime_usec());
    vftr_print_profile (vftr_log, f_html, &ntop, vftr_get_runtime_usec());
#ifdef _MPI
    ////if (vftr_environment.print_stack_profile->value) {
       // Inside of vftr_print_function_statistics, we use an MPI_Allgather to compute MPI imbalances. Therefore,
       // we need to call this function for every rank, but give it the information of vftr_profile_wanted
       // to avoid unrequired output.
       vftr_print_function_statistics (vftr_log, display_functions, n_display_functions, vftr_profile_wanted);
    ////}
#endif
 
    funcTable = vftr_func_table;

    if (vftr_profile_wanted) {
        vftr_print_global_stacklist(vftr_log);
    }

    vftr_finalize_vfd_file (finalize_time, vftr_signal_number);
    if (vftr_events_enabled && vftr_stop_hwc() < 0) {
	fprintf(vftr_log, "error stopping H/W counters, ignored\n");
    }

    if (vftr_max_allocated_fields > 0) vftr_allocate_finalize(vftr_log);

    if (vftr_environment.print_env->value) vftr_print_environment(vftr_log);
    
    if (vftr_log) {
    	bool is_empty = (ftello (vftr_log) == (off_t)0);
    	if (is_empty) unlink (vftr_logfile_name);

    	fclose (vftr_log);
    }
    vftr_switch_off();
}

/**********************************************************************/

// vftr_finalize has to be called in the wrapper of MPI_Finalize, both for C and Fortran.
// This is the corresponding symbol for Fortran, with an added "_".
void vftr_finalize_() {
	vftr_finalize();
#ifdef _MPI
	PMPI_Barrier (MPI_COMM_WORLD);
#endif
}

/**********************************************************************/

int vftr_setup_test_1 (FILE *fp) {
	fprintf (fp, "Check MPI rank and size received from environment variables\n");		
	int mpi_rank, mpi_size;
	vftr_get_mpi_info (&mpi_rank, &mpi_size);
	fprintf (fp, "Rank: %d\n", mpi_rank);
	fprintf (fp, "Size: %d\n", mpi_size);
	return 0;
}

/**********************************************************************/

int vftr_setup_test_2 (FILE *fp) {
	fprintf (fp, "Check disclaimers\n");
	vftr_print_disclaimer_full (fp);
	fprintf (fp, "****************************************\n");
	vftr_print_disclaimer (fp);
	return 0;
}

