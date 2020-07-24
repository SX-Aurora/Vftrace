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

#include <assert.h>
#include <string.h>
#include <signal.h>

#include "vftr_symbols.h"
#include "vftr_hwcounters.h"
#include "vftr_omp.h"
#include "vftr_setup.h"
#include "vftr_environment.h"
#include "vftr_hooks.h"
#include "vftr_dlopen.h"
#include "vftr_filewrite.h"
#include "vftr_functions.h"
#include "vftr_pause.h"
#include "vftr_timer.h"
#include "vftr_stacks.h"
#include "vftr_dlopen.h"
#include "vftr_async_messages.h"

bool vftr_profile_wanted = false;

int vftr_compare (const void *a1, const void *a2) {
    function_t *f1 = *(function_t **)a1;
    function_t *f2 = *(function_t **)a2;
    if(!f2) return -1; /* Order important to work around SX qsort problem */
    if(!f1) return  1;
    long long t1 = f1->prof_current[0].timeExcl - f1->prof_previous[0].timeExcl;
    long long t2 = f2->prof_current[0].timeExcl - f2->prof_previous[0].timeExcl;
    long long diff = t2 - t1;
    if( diff > 0 ) return  1;
    if( diff < 0 ) return -1;
    return  0;
}

/**********************************************************************/

void vftr_save_old_state (int me) {
    int i,j;

    for (j = 0; j < vftr_stackscount; j++) {
        function_t *func = vftr_func_table[j];
        func->prof_previous[me].calls  = func->prof_current[me].calls;
        func->prof_previous[me].cycles = func->prof_current[me].cycles;
        func->prof_previous[me].timeExcl = func->prof_current[me].timeExcl;
        for (i = 0; i < vftr_n_hw_obs; i++) {
            func->prof_previous[me].event_count[i] = func->prof_current[me].event_count[i];
	}
    }
}

/**********************************************************************/

void vftr_function_entry (const char *s, void *addr, int line, bool isPrecise) {
    int e, me, read_counters;
    unsigned long long timer, time0, delta;
    unsigned long long cycles0;
    double wtime;
    static int vftr_init = 1;
    bool time_to_sample = false;
    function_t *caller, *func, *callee;
    profdata_t *prof_return;

    if (vftr_init) {
	vftr_initialize ();
	vftr_init = 0;
    }
    if (vftr_off() || vftr_paused) return;

    long long func_entry_time = vftr_get_runtime_usec();
    // log function entry and exit time to estimate the overhead time
    long long overhead_time_start = func_entry_time;
    me = OMP_GET_THREAD_NUM;
    timer = vftr_get_runtime_usec ();
    time0 = timer - vftr_inittime;
    cycles0 = vftr_get_cycles() - vftr_initcycles;

    // This is the hook for shared libraries opened during the
    // application's runtime. The preloaded library vftr_dlopen.so
    // triggers this flag, which leads to a renewed creation of the
    // symbol table, containing the symbols of the dlopened library
    //
    if (lib_opened) {
	lib_opened = 0;
    	vftr_create_symbol_table (vftr_mpirank, NULL);
    }

    caller = vftr_fstack[me];
    // If the caller address equals the current address, we are
    // dealing with (simple) recursive function call. We filter
    // them out to avoid extensively large call stacks, e.g. 
    // in the viewer. 
    // In principle, we could just return from this function
    // in the given situation. However, the same stack is encountered
    // in the function exit as well, and its treatment needs to be 
    // avoided there too. This is easily solved by introducing the
    // counter "recursion_depth". For each recursive function entry,
    // it is incremented, for each recursive function exit it is decremented.
    // This way, in vftr_function_exit, when it is zero we know that
    // we are not dealing with a recursive function. 
    // 
    if (addr == caller->address) {
        caller->prof_current[me].calls++;
	caller->recursion_depth++;
        return;
    }

#ifdef _OPENMP
#pragma omp critical
#endif
{
    //
    // Check if the function is in the table
    //
    callee = caller->call;
    
    if (callee == NULL) {
        // No calls at all yet: add new function
        func = vftr_new_function(addr, s, caller, "no calls yet", line, isPrecise);
    } else {
	// Search the function name in the function list
        func = callee;
        if (func->address != addr) {
           for ( ;; ) {
               func = func->next;
               if (func == callee || func->address == addr) {
           	break;
               }
           }
           if (func == callee) {
               // No call from this callee yet: add new function
               func = vftr_new_function(addr, s, caller, "not in list", line, isPrecise);
           }
        }
    }
    caller->call = func; // Faster lookup next time around
}

    if (func->exclude_this) return;
    if (line > 0) assert (func->line_beg == line);

    if (func->profile_this) {
        wtime = (vftr_get_runtime_usec() - vftr_overhead_usec[me]) * 1.0e-6;
        vftr_print_stack (me, wtime, func, "profile before call to", 0);
        vftr_profile_wanted = true;
        int ntop;
        vftr_print_profile (vftr_log, &ntop, time0);
        vftr_print_local_stacklist (vftr_func_table, vftr_log, ntop);
	vftr_save_old_state (me);
    }

    vftr_fstack[me] = func; /* Here's where we are now */

    // Is it time for the next sample?
    time_to_sample = (func_entry_time > vftr_nextsampletime[me]) || func->precise;  

    read_counters = (func->ret->detail || func->detail) &&
		    vftr_events_enabled && 
                    (time_to_sample || vftr_environment->accurate_profile->value);
#ifdef __ve__
    // Do not read performance counters for OpenMP threads > 0
    if (me) read_counters = 0;
#endif

    if (time_to_sample && vftr_env_do_sampling ()) {
        vftr_write_to_vfd (func_entry_time, vftr_prog_cycles[me], func->id, SID_ENTRY, me);
#ifdef _MPI
        int mpi_isinit;
        PMPI_Initialized(&mpi_isinit);
        if (mpi_isinit) {
           int mpi_isfinal;
           PMPI_Finalized(&mpi_isfinal);
           if (!mpi_isfinal) {
              vftr_clear_completed_request();
           }
        }
#endif
    }

    func->prof_current[me].calls++;

    // Maintain profile

    if (func->ret && func->ret->prof_current) {
        prof_return = &func->ret->prof_current[me];
        delta = cycles0 - vftr_prof_data[me].cycles;
	prof_return->cycles += delta;
        prof_return->timeExcl += func_entry_time - vftr_prof_data[me].timeExcl;
        vftr_prog_cycles[me] += delta;
        func->prof_current[me].timeIncl -= func_entry_time;
	if (read_counters) {
            int ic = vftr_prof_data[me].ic;
            vftr_read_counters (vftr_prof_data[me].events[ic], me);
            if (prof_return->event_count && func->ret->detail) {
                for (e = 0; e < vftr_n_hw_obs; e++) {
                    long long delta = vftr_prof_data[me].events[ic][e] - vftr_prof_data[me].events[1-ic][e];
#ifdef __ve__
                    if (delta < 0) /* Handle counter overflow */
                        delta += e < 2 ? (long long) 0x000fffffffffffff
                                       : (long long) 0x00ffffffffffffff;
#endif
		    prof_return->event_count[e] += delta;
                }
            }
	    vftr_prof_data[me].ic = 1 - ic;
	}
    }

    /* Compensate overhead */
    // The stuff we did here added up cycles. Therefore, we have to reset
    // the global cycle count and time value.
    vftr_prof_data[me].cycles = vftr_get_cycles() - vftr_initcycles;
    long long overhead_time_end = vftr_get_runtime_usec();
    vftr_prof_data[me].timeExcl = overhead_time_end;
    vftr_overhead_usec[me] += overhead_time_end - overhead_time_start;
}

/**********************************************************************/

void vftr_function_exit(int line) {
    int           e, me, read_counters, timeToSample;
    long long     time0, timer;
    unsigned long long cycles0;
    function_t    *func;
    double        wtime;
    profdata_t *prof_current;

    if (vftr_off() || vftr_paused) return;

    me = OMP_GET_THREAD_NUM;
    /* See at the beginning of vftr_function_entry: If
     * we are dealing with a recursive function call, exit.
     */
    if (vftr_fstack[me]->recursion_depth) {
        vftr_fstack[me]->recursion_depth--;
        return;
    }
    long long func_exit_time = vftr_get_runtime_usec();
    // get the time to estimate vftrace overhead
    long long overhead_time_start = func_exit_time;

    timer = vftr_get_runtime_usec ();
    time0 = timer - vftr_inittime;
    cycles0 = vftr_get_cycles() - vftr_initcycles;
    func  = vftr_fstack[me];
    if (func->exclude_this) return;

    if (line > 0) {
        if (func->line_end) {
		assert (func->line_end == line);
        } else {
            /* Complete region name */
            char *endline = index (func->name, '-'); // string.h's index function
            snprintf (endline+1, 7, "%d", line);
            func->line_end = line;
        }
    }

    prof_current = &func->prof_current[me];
    prof_current->timeIncl += func_exit_time;   /* Inclusive time */
    
    vftr_fstack[me] = func->ret;

    /* Is it time for the next sample? */

    timeToSample = (func_exit_time > vftr_nextsampletime[me]) || func->precise || 
                   (func->ret && !func->ret->id) /* Return from main program: end of execution */;

    read_counters = (func->ret->detail || func->detail) &&
	  	    vftr_events_enabled && 
                    (timeToSample || vftr_environment->accurate_profile->value);
#ifdef __ve__
    if (me) read_counters = 0;
#endif

    if (timeToSample && vftr_env_do_sampling ()) {
        vftr_write_to_vfd(func_exit_time, prof_current->cycles, func->id, SID_EXIT, me);
#ifdef _MPI
        int mpi_isinit;
        PMPI_Initialized(&mpi_isinit);
        if (mpi_isinit) {
           int mpi_isfinal;
           PMPI_Finalized(&mpi_isfinal);
           if (!mpi_isfinal) {
              vftr_clear_completed_request();
           }
        }
#endif
    }

    /* Maintain profile info */

    prof_current->cycles += cycles0;
    prof_current->timeExcl += func_exit_time;
    vftr_prog_cycles[me] += cycles0;
    if (func->ret) {
        prof_current->cycles -= vftr_prof_data[me].cycles;
        prof_current->timeExcl -= vftr_prof_data[me].timeExcl;
        vftr_prog_cycles[me] -= vftr_prof_data[me].cycles;
    }

    if (read_counters) {
        int ic = vftr_prof_data[me].ic;
        vftr_read_counters (vftr_prof_data[me].events[ic], me);
        prof_current->ecreads++; /* Only at exit */
        if (prof_current->event_count && func->detail) {
            for (e = 0; e < vftr_n_hw_obs; e++) {
                long long delta = vftr_prof_data[me].events[ic][e] - vftr_prof_data[me].events[1-ic][e];
#ifdef __ve__
	        /* Handle counter overflow */
                if (delta < 0) {
                        delta += e < 2 ? (long long) 0x000fffffffffffff
                                       : (long long) 0x00ffffffffffffff;
 		}
#endif
		prof_current->event_count[e] += delta;
            }
        }
        vftr_prof_data[me].ic = 1 - ic;
    }

    wtime = (vftr_get_runtime_usec() - vftr_overhead_usec[me]) * 1.0e-6;

    if (func->profile_this)  {
        vftr_print_stack (me, wtime, func, "profile at exit from", timeToSample);
        vftr_profile_wanted = true;
        int ntop;
        vftr_print_profile (stdout, &ntop, time0);
        vftr_print_local_stacklist( vftr_func_table, stdout, ntop );
    }

    if (me == 0 && time0 >= vftr_timelimit) {
        fprintf( vftr_log, "vftr_timelimit exceeded - terminating execution\n" );
	kill( getpid(), SIGTERM );
    }
#if defined(_SX) || defined(__ve__OFF)
#ifdef _OPENMP
    if( omp_in_parallel() ) vftr_in_parallel[me]--;
#endif
#endif

    /* Sort profile if it is time */
    
    if (me == 0 && wtime >= vftr_sorttime)  {
        int i, top;
        double tsum = 0.;
        double scale = 100. / (double)vftr_prog_cycles[me];

        /* TO DO for OpenMP: set lock guarding vftr_new_function() usage */

        qsort (vftr_func_table, (size_t)vftr_stackscount, sizeof( function_t *),
	       vftr_compare);

        /* Set function detail flags while sum(time) < max */
        for (i = 0; i < vftr_stackscount; i++) {
            function_t *f = vftr_func_table[i];
            tsum += (double)f->prof_current[0].cycles;
	    double cutoff = vftr_environment->detail_until_cum_cycles->value;
            if ((tsum * scale) > cutoff) break;
            f->detail = 1;
        }
        top = i;
        /* Clear function detail flags for all others */
        for(; i<vftr_stackscount; i++) 
            vftr_func_table[i]->detail = 0;

        /* TO DO for OpenMP: unset lock guarding vftr_new_function() usage */

        vftr_sorttime *= vftr_sorttime_growth;
    }

    /* Compensate overhead */
    // The stuff we did here added up cycles. Therefore, we have to reset
    // the global cycle count and time value.
    vftr_prof_data[me].cycles = vftr_get_cycles() - vftr_initcycles;
    long long overhead_time_end = vftr_get_runtime_usec();
    vftr_prof_data[me].timeExcl = overhead_time_end;
    vftr_overhead_usec[me] += overhead_time_end - overhead_time_start;
}

// These are the actual Cygnus function hooks. 
//
#if defined(__x86_64__) || defined(__ve__)

void __cyg_profile_func_enter (void *func, void *caller) {
    vftr_function_entry (NULL, func, -1, false);
}

void __cyg_profile_func_exit (void *func, void *caller) {
    vftr_function_exit( -1 );
}

#elif defined(__ia64__)

// The argument func is a pointer to a pointer instead of a pointer.

void __cyg_profile_func_enter (void **func, void *caller) {
    vftr_function_entry( NULL, *func, -1, false);
}

void __cyg_profile_func_exit  (void **func, void *caller) {
    vftr_function_exit( -1 );
}
#endif
