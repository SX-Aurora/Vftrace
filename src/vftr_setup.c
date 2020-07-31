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
#include "vftr_omp.h"
#include "vftr_environment.h"
#include "vftr_setup.h"
#include "vftr_dlopen.h"
#include "vftr_timer.h"
#include "vftr_fileutils.h"
#include "vftr_filewrite.h"
#include "vftr_signals.h"
#include "vftr_stacks.h"
#include "vftr_hooks.h"
#include "vftr_loadbalance.h"
#include "vftr_timer.h"

bool vftr_timer_end;

int vftr_mpirank;
int vftr_mpisize;
// Indicates if Vftrace is in an OMP parallel region
int *vftr_in_parallel;
unsigned int *vftr_samplecount;

void vftr_print_disclaimer_full () {
    fprintf( vftr_log, 
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

void vftr_print_disclaimer () {
    int v_major = MAJOR_VERSION;
    int v_minor = MINOR_VERSION;
    int rev = REVISION;
    fprintf (vftr_log, "Vftrace version %d.%d.%d\n", v_major, v_minor, rev);
    fprintf (vftr_log, "Runtime profile for application: %s\n", "");
    fprintf (vftr_log, "Date: "); 
    fprintf( vftr_log, 
        "This is free software with ABSOLUTELY NO WARRANTY.\n"
        "For details: use vftrace with environment variable VFTR_LICENSE\n"
        "set to 1, or run \"vfview -w\", or consult the COPYRIGHT file.\n" );
}

/**********************************************************************/

#ifdef _OPENMP
void vftr_init_omp_locks () {
    omp_init_lock (&vftr_lock);
    omp_init_lock (&vftr_lock_exp);
    omp_init_lock (&vftr_lock_hook);
    omp_init_lock (&vftr_lock_prof);
}
#endif

/**********************************************************************/

// Assuming vftr_initialize will be called outside parallel region
void vftr_initialize() {
    char *s;
    char *logfile_nameformat;
    int j, n, me;
    int task_digits, thread_digits;

    me = OMP_GET_THREAD_NUM;

    // set the timer reference point for this process
    vftr_set_local_ref_time();
    
    // measure overhead induced by vftrace
    long long overhead_time_start = vftr_get_runtime_usec();

    vftr_read_environment ();
    if (vftr_off()) {
	return;
    }
    vftr_assert_environment ();
	
    lib_opened = 0;

    char *vftr_program_path;
    vftr_timelimit = LONG_MAX;

    // No buffering for messages going directly to stdout
    setvbuf (stdout, NULL, _IOLBF, (size_t)0);

    sprintf (vftr_fileid, "VFTRACE %07d", 
	(100 * MAJOR_VERSION + MINOR_VERSION) * 100 + REVISION);

    logfile_nameformat = malloc(1024*sizeof(char));

    vftr_omp_threads = OMP_GET_MAX_THREADS;
    vftr_samplecount = (unsigned int *) malloc (vftr_omp_threads * sizeof(unsigned int));
    vftr_prog_cycles = (long long *) malloc (vftr_omp_threads * sizeof(long long));
    vftr_in_parallel = (int *) malloc (vftr_omp_threads * sizeof(unsigned int));
    assert (vftr_samplecount);

    vftr_overhead_usec = (long long *) malloc (vftr_omp_threads * sizeof(long long));
    assert (vftr_overhead_usec);
    for (int i = 0; i < vftr_omp_threads; i++) {
	vftr_overhead_usec[i] = 0ll;
    }
    assert (vftr_prog_cycles);
    assert (vftr_in_parallel);

    memset (vftr_prog_cycles, 0, vftr_omp_threads * sizeof(long long));

#ifdef _OPENMP
   vftr_init_omp_locks ();
#endif

    vftr_program_path = get_application_name ();
    char *basename;
    if (vftr_environment->logfile_basename->set) {
	basename = vftr_environment->logfile_basename->value;
    } else {
	char *s;	
	if (s = rindex (vftr_program_path, '/')) {
		basename = strdup (s + 1);
	} else {
		basename = strdup (vftr_program_path);
	}	
    }

#ifdef _MPI
    if (getenv ("PMI_RANK")) {
        vftr_mpirank  = atoi( getenv( "PMI_RANK" ) );
        vftr_mpisize  = atoi( getenv( "PMI_SIZE" ) );
    } else if( getenv( "OMPI_COMM_WORLD_RANK" ) ) {
        vftr_mpirank  = atoi( getenv( "OMPI_COMM_WORLD_RANK" ) );
	vftr_mpisize  = atoi( getenv( "OMPI_COMM_WORLD_SIZE" ) );
    } else if( getenv( "PMI_ID" ) ) {
        vftr_mpirank  = atoi( getenv( "PMI_ID" ) );
	vftr_mpisize  = atoi( getenv( "MPIRUN_NPROCS" ) );
    } else if( getenv( "MPIRUN_RANK" ) ) {
        vftr_mpirank  = atoi( getenv( "MPIRUN_RANK" ) );
	vftr_mpisize  = atoi( getenv( "MPIRUN_NPROCS" ) );
    } else if( getenv( "MPIRANK" ) ) {
        vftr_mpirank  = atoi( getenv( "MPIRANK" ) );
        /* MPISIZE not set by MPI/SX, will be set after mpi_init call */
	vftr_mpisize  = atoi( getenv( "MPISIZE" ) );
    } else if( getenv( "SLURM_PROCID"   ) ) {
	vftr_mpirank  = atoi( getenv( "SLURM_PROCID" ) );
	if( s = getenv( "SLURM_NPROCS" ) ) vftr_mpisize  = atoi( s );
    } else {
	// Cannot find out how many MPI ranks there are, assume only one.
	vftr_mpirank = 0;
	vftr_mpisize = 1;
    }
#else
    // No MPI, only one rank exists.
    vftr_mpirank = 0;
    vftr_mpisize = 1;
#endif


    // Count the number of digits in the number of MPI tasks and OpenMP threads
    // in order to have the correct format in the log file name.
    task_digits = count_digits (vftr_mpisize);
    thread_digits = count_digits (vftr_omp_threads);

    // Build filename format in a form that the filenames will
    // be sorted in the correct way
    //
    sprintf (logfile_nameformat, "%s/%s_%%0%dd.log",
             vftr_environment->output_directory->value,
             basename, task_digits);
    sprintf (vftr_logfile_name, logfile_nameformat, vftr_mpirank);
    vftr_log = fopen( vftr_logfile_name, "w+");
    assert (vftr_log);
    
    // Do not buffer when writing into the log file
    setvbuf (vftr_log, NULL, _IOLBF, (size_t)0);

#ifndef _OPENMP
    if (vftr_omp_threads > 1 && !vftr_mpirank) {
        fprintf( vftr_log, 
	  "WARNING: vftrace library used does not support OpenMP\n" );
    }
#endif

    if (!vftr_mpirank) {
       if (vftr_environment->license_verbose->value) {
	  vftr_print_disclaimer_full ();
       } else {
	  vftr_print_disclaimer ();
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


    /* Formating info for call tree */
    vftr_maxtime = (long long *) malloc (vftr_omp_threads * sizeof(long long));

    n = vftr_omp_threads * sizeof(profdata_t);
    vftr_prof_data = (profdata_t *) malloc( n );
    assert (vftr_prof_data);

    memset (vftr_prof_data, 0, n);

    // initialize the stack variables and tables
    vftr_initialize_stacks();

    /* Allocate file pointers for each thread */
    vftr_vfd_file = (FILE **) malloc( vftr_omp_threads * sizeof(FILE *) );
    assert (vftr_vfd_file);
    
    /* Allocate time arrays for each thread */
    n = vftr_omp_threads * sizeof(long long);
    vftr_nextsampletime = (long long *) malloc (n) ;
    vftr_prevsampletime = (long long *) malloc (n) ;
    assert (vftr_nextsampletime);
    assert (vftr_prevsampletime);
    for (int i = 0; i < vftr_omp_threads; i++) {
        vftr_nextsampletime[i] = 0;
        vftr_prevsampletime [i] = 0;
    }

#ifdef _OPENMP
    /* Regular expressions to detect OpenMP regions */
    vftr_openmpregexp = vftr_compile_regexp( "\\$[0-9][0-9_]*$" );
#endif

    /* Init event counters */
    vftr_n_hw_obs = 0;
    if (vftr_environment->scenario_file->set) {
	vftr_events_enabled = !vftr_init_hwc (vftr_environment->scenario_file->value);
    } else {
	vftr_events_enabled = false;
    }
    if (vftr_n_hw_obs < 0) {
        fprintf(vftr_log, "error initializing H/W counters\n");
        vftr_events_enabled = false;
    }

    if (vftr_n_hw_obs  > 0) {
    	for (int i = 0; i < vftr_omp_threads; i++) {
    	    vftr_prof_data[i].events[0] = (long long *) malloc (vftr_n_hw_obs * sizeof(long long));
    	    vftr_prof_data[i].events[1] = (long long *) malloc (vftr_n_hw_obs * sizeof(long long));
    	    memset (vftr_prof_data[i].events[0], 0, vftr_n_hw_obs * sizeof(long long));
    	    memset (vftr_prof_data[i].events[1], 0, vftr_n_hw_obs * sizeof(long long));
    	}
    }

    vftr_inittime = vftr_get_runtime_usec ();
    vftr_initcycles = vftr_get_cycles();
    // convert the sampletime and timelimit to microseconds
    vftr_interval  = (long long) (vftr_environment->sampletime->value * 1.0e6);
    assert (vftr_interval > 0ll);
    vftr_timelimit = vftr_environment->stoptime->value * 1000000ll;

    vftr_sorttime = 15.;
    vftr_sorttime_growth = 2.;

    /* Create VFD files */
    if (vftr_env_do_sampling ()) {
	vftr_init_vfd_file (basename, task_digits, thread_digits);
    }
    
    vftr_profile_wanted = (vftr_environment->logfile_all_ranks->value) ||
                          (vftr_mpirank == 0);

    if (vftr_environment->print_stacks_for->set) {
        char *vftr_print_groups = vftr_environment->print_stacks_for->value;
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
    if (!vftr_environment->signals_off->value) {
	vftr_define_signal_handlers ();
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    
    fflush (stdout);
    vftr_inittime = vftr_get_runtime_usec (); /* Will be updated later if MPI used */
    vftr_initcycles = vftr_get_cycles();
    
    // get the time to estimate vftrace overhead
    long long overhead_time_end = vftr_get_runtime_usec();
    vftr_overhead_usec[me] += overhead_time_end - overhead_time_start;
}

/**********************************************************************/

void vftr_calc_tree_format (function_t *func) {
    function_t *f;
    int         me, i, n;
    long long   fcalls, ftime;

    if (func == NULL) return;

    for (me = 0; me < vftr_omp_threads; me++) {
        fcalls = func->prof_current[me].calls;
        ftime  = func->prof_current[me].cycles;
        if (vftr_maxtime[me]  < ftime) vftr_maxtime [me] = ftime;
    }

    n = func->levels;

    /* Recursive search of callees */
    for (i = 0, f = func->first; i < n; i++,f = f->next) {
        vftr_calc_tree_format (f);
    }
}

/**********************************************************************/

void vftr_finalize() {
    int            i, me, ntop = 0;
    function_t     **funcTable;

    if (vftr_off())  return;

    // get the total runtime
    long long finalize_time = vftr_get_runtime_usec();
    long long timer = vftr_get_runtime_usec ();
    long long time0 = timer - vftr_inittime;

    vftr_timer_end = true;

    // Mark end of non-parallel interval
    if (vftr_env_do_sampling()) {
        for (me = 1; me < vftr_omp_threads; me++)
            vftr_write_to_vfd (finalize_time, vftr_prog_cycles[me], 0, SID_EXIT, me);
    }
    
    bool valid_loadbalance_table = !vftr_normalize_stacks();
    vftr_calc_tree_format (vftr_froots[0]);

    vftr_print_profile (vftr_log, &ntop, time0);

    funcTable = vftr_func_table;

    callsTime_t **loadbalance_info;
    if (valid_loadbalance_table) {
	loadbalance_info = vftr_get_loadbalance_info( funcTable );
    }

    bool is_parallel = vftr_mpisize > 1 || vftr_omp_threads > 1;
    if (is_parallel && vftr_mpirank == 0 && valid_loadbalance_table) {
        int *loadIDs = (int *) malloc (vftr_gStackscount * sizeof(int));
        int nLoadIDs;
	int group_base, group_size;
        if (vftr_environment->print_loadinfo_for->set) {
	    char *vftr_mpi_groups = vftr_environment->print_loadinfo_for->value;
            while (*vftr_mpi_groups) {
                char *p;
		// Loop to the end of the group, indicated by ","
                for (p = vftr_mpi_groups; *p && *p != ','; p++);
                sscanf (vftr_mpi_groups, "%d:%d", &group_base, &group_size);
		// Is it guaranteed to be <= 0 if no group size is given? 
                if (group_size <= 0) break;
                vftr_print_loadbalance (loadbalance_info, group_base, group_size, vftr_log,
                                        loadIDs, &nLoadIDs);
                vftr_print_global_stacklist (vftr_log);
	        // Check if there is anything behind the comma of the environment variable. If so, proceed.
                vftr_mpi_groups = *p ? p + 1 : p;
            }
        } else {
            vftr_print_loadbalance (loadbalance_info, 0, vftr_mpisize, vftr_log, loadIDs, &nLoadIDs);
            vftr_print_global_stacklist (vftr_log);
        }
        if (valid_loadbalance_table) free (*loadbalance_info);
    }

    if (vftr_profile_wanted) {
    //if (vftr_profile_wanted && valid_loadbalance_table) {
        vftr_print_global_stacklist(vftr_log);
        vftr_print_local_demangled( vftr_func_table, vftr_log, ntop );
    }

    vftr_finalize_vfd_file (finalize_time, vftr_signal_number);
    if (vftr_events_enabled && vftr_stop_hwc() < 0) {
	fprintf(vftr_log, "error stopping H/W counters, ignored\n");
    }
    
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
