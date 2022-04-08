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

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdbool.h>
#include "vftr_symbols.h"
#include "vftr_hwcounters.h"
#include "vftr_setup.h"
#include "vftr_environment.h"
#include "vftr_hooks.h"
#include "vftr_dlopen.h"
#include "vftr_filewrite.h"
#include "vftr_functions.h"
#include "vftr_pause.h"
#include "vftr_timer.h"
#include "vftr_stacks.h"
#include "clear_mpi_requests.h"
#include "vftr_sorting.h"
#include "vftr_mallinfo.h"
#include "vftr_cuda.h"

bool vftr_profile_wanted = false;

/**********************************************************************/

void vftr_flush_cuda_events_to_func (function_t *func) {
    cuda_event_list_t *cuda_events;
    vftr_cuda_flush_events (&cuda_events);
    if (cuda_events != NULL) {
      if (func->cuda_events == NULL) {
         func->cuda_events = cuda_events; 
      } else {
         cuda_event_list_t *t1 = cuda_events;
         while (t1 != NULL) {
            cuda_event_list_t *t2 = func->cuda_events;
            while (true) {
               // The function obtained from vftr_cuda has not been registered for this function yet.
               if (t2 == NULL) {
                  t2 = (cuda_event_list_t*) malloc (sizeof(cuda_event_list_t));
                  t2->next = NULL;
                  t2->func_name = t1->func_name;
                  t2->t_acc[T_CUDA_COMP] = t1->t_acc[T_CUDA_COMP];
                  t2->t_acc[T_CUDA_MEMCP] = t1->t_acc[T_CUDA_MEMCP];
                  t2->n_calls = t1->n_calls; 
                  break;
               } else if (!strcmp (t1->func_name, t2->func_name)) {
                  t2->t_acc[T_CUDA_COMP] += t1->t_acc[T_CUDA_COMP];
                  t2->t_acc[T_CUDA_MEMCP] += t1->t_acc[T_CUDA_MEMCP];
                  t2->n_calls += t1->n_calls;
                  break;
               } 
               t2 = t2->next;
            }      
            t1 = t1->next;
         }
         free(cuda_events);
      }
    } else {
      //printf ("func: %s - no CUDA events\n", func->name);
    }
}

/**********************************************************************/

void vftr_print_stack_at_runtime (function_t *this_func, bool is_entry, bool time_to_sample) {
   vftr_prof_times_t prof_times = vftr_get_application_times_all (vftr_get_runtime_usec());
   char *msg = is_entry ? "profile before call to" : "profile_at_exit_from"; 
   
   vftr_write_stack_ascii (vftr_log, prof_times.t_sec[TOTAL_TIME] - prof_times.t_sec[TOTAL_OVERHEAD],
                           this_func, msg, time_to_sample);
   int n_functions_top = vftr_count_func_indices_up_to_truncate (vftr_func_table,
                   prof_times.t_usec[TOTAL_TIME] - prof_times.t_usec[SAMPLING_OVERHEAD]);
   // No sorting of the function table at this point. We want a snapshot of the current state.
   vftr_print_profile (stdout, vftr_func_table, n_functions_top, prof_times, 0, NULL);
   vftr_print_local_stacklist (vftr_func_table, vftr_log, n_functions_top);
}

/**********************************************************************/

void vftr_function_entry (const char *s, void *addr, bool isPrecise) {
    bool time_to_sample, read_counters;
    unsigned long long delta, cycles0;
    static bool vftr_needs_init = true;
    function_t *caller, *func, *callee;
    profdata_t *prof_return;

    if (vftr_needs_init) {
	vftr_initialize ();
	vftr_needs_init = 0;
    }
    if (vftr_off() || vftr_paused) return;
#ifdef _OPENMP
    if (omp_get_thread_num() > 0) return;
#endif

    long long func_entry_time = vftr_get_runtime_usec();
    // log function entry and exit time to estimate the overhead time
    long long overhead_time_start = func_entry_time;
    cycles0 = vftr_get_cycles() - vftr_initcycles;

    // This is the hook for shared libraries opened during the
    // application's runtime. The preloaded library vftr_dlopen.so
    // triggers this flag, which leads to a renewed creation of the
    // symbol table, containing the symbols of the dlopened library
    //
    if (lib_opened) {
	lib_opened = 0;
    	vftr_create_symbol_table (vftr_mpirank);
    }

    caller = vftr_fstack;
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
        caller->prof_current.calls++;
	caller->recursion_depth++;
        return;
    }

    if (vftr_n_cuda_devices > 0) vftr_flush_cuda_events_to_func (caller);

    //
    // Check if the function is in the table
    //
    callee = caller->callee;
    
    if (callee == NULL) {
        // No calls at all yet: add new function
        func = vftr_new_function(addr, s, caller, isPrecise);
    } else {
	// Search the function address in the function list
        func = callee;
        if (func->address != addr) {
           for ( ;; ) {
               func = func->next_in_level;
               if (func == callee || func->address == addr) {
           	  break;
               }
           }
           if (func == callee) {
               // No call from this callee yet: add new function
               func = vftr_new_function(addr, s, caller, isPrecise);
           }
        }
    }
    caller->callee = func; // Faster lookup next time around
    vftr_fstack = func; /* Here's where we are now */
    func->open = true;

    if (func->profile_this) {
        vftr_print_stack_at_runtime (func, true, false);
        vftr_profile_wanted = true;
    }


    // Is it time for the next sample?
    time_to_sample = (func_entry_time > vftr_nextsampletime) || func->precise;  

    read_counters = (func->return_to->detail || func->detail) &&
		    vftr_events_enabled && 
                    (time_to_sample || vftr_environment.accurate_profile->value);

    if (func->return_to) {
        prof_return = &func->return_to->prof_current;
        if (read_counters) {
           int ic = vftr_prof_data.ic;
           vftr_read_counters (vftr_prof_data.events[ic]);
           if (prof_return->event_count && func->return_to->detail) {
               for (int e = 0; e < vftr_n_hw_obs; e++) {
                   long long delta = vftr_prof_data.events[ic][e] - vftr_prof_data.events[1-ic][e];
#ifdef __ve__
                   if (delta < 0) /* Handle counter overflow */
                       delta += e < 2 ? (long long) 0x000fffffffffffff
                                      : (long long) 0x00ffffffffffffff;
#endif
    	           prof_return->event_count[e] += delta;
               }
           }
           vftr_prof_data.ic = 1 - ic;
       }

    }

    profdata_t *prof_current = &func->prof_current;
    if (vftr_memtrace) {
       vftr_sample_vmrss (prof_current->calls, true, false, prof_current->mem_prof);
    }

    if (time_to_sample && vftr_env_do_sampling ()) {
        vftr_write_to_vfd (func_entry_time, prof_current, func->id, SID_ENTRY);
#ifdef _MPI
        int mpi_isinit;
        PMPI_Initialized(&mpi_isinit);
        if (mpi_isinit) {
           int mpi_isfinal;
           PMPI_Finalized(&mpi_isfinal);
           if (!mpi_isfinal) {
              vftr_clear_completed_requests();
           }
        }
#endif
    }

    func->prof_current.calls++;

    // Maintain profile

    if (func->return_to) {
        delta = cycles0 - vftr_prof_data.cycles;
	prof_return->cycles += delta;
        prof_return->time_excl += func_entry_time - vftr_prof_data.time_excl;
        vftr_prog_cycles += delta;
        func->prof_current.time_incl -= func_entry_time;
    }

    /* Compensate overhead */
    // The stuff we did here added up cycles. Therefore, we have to reset
    // the global cycle count and time value.
    vftr_prof_data.cycles = vftr_get_cycles() - vftr_initcycles;
    long long overhead_time_end = vftr_get_runtime_usec();
    vftr_prof_data.time_excl = overhead_time_end;
    vftr_overhead_usec += overhead_time_end - overhead_time_start;
    func->overhead += overhead_time_end - overhead_time_start;
}

/**********************************************************************/

void vftr_function_exit () {
    int           e;
    bool time_to_sample, read_counters;
    long long timer;
    unsigned long long cycles0;
    function_t *func;
    double wall_time;
    profdata_t *prof_current;

    if (vftr_off() || vftr_paused) return;
#ifdef _OPENMP
    if (omp_get_thread_num() > 0) return;
#endif

    /* See at the beginning of vftr_function_entry: If
     * we are dealing with a recursive function call, exit.
     */
    if (vftr_fstack->recursion_depth) {
        vftr_fstack->recursion_depth--;
        return;
    }
    long long func_exit_time = vftr_get_runtime_usec();
    // get the time to estimate vftrace overhead
    long long overhead_time_start = func_exit_time;

    timer = vftr_get_runtime_usec ();
    cycles0 = vftr_get_cycles() - vftr_initcycles;
    func  = vftr_fstack;

    if (vftr_n_cuda_devices > 0) vftr_flush_cuda_events_to_func (func);

    prof_current = &func->prof_current;
    prof_current->time_incl += func_exit_time;   /* Inclusive time */
    
    vftr_fstack = func->return_to;

    /* Is it time for the next sample? */

    time_to_sample = (func_exit_time > vftr_nextsampletime) || func->precise || 
                   (func->return_to && !func->return_to->id) /* Return from main program: end of execution */;

    read_counters = (func->return_to->detail || func->detail) &&
	  	    vftr_events_enabled && 
                    (time_to_sample || vftr_environment.accurate_profile->value);

    if (read_counters) {
        int ic = vftr_prof_data.ic;
        vftr_read_counters (vftr_prof_data.events[ic]);
        if (prof_current->event_count && func->detail) {
            for (e = 0; e < vftr_n_hw_obs; e++) {
                long long delta = vftr_prof_data.events[ic][e] - vftr_prof_data.events[1-ic][e];
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
        vftr_prof_data.ic = 1 - ic;
    }

    if (vftr_memtrace) {
       vftr_sample_vmrss (prof_current->calls - 1, false, false, prof_current->mem_prof);
    }

    if (time_to_sample && vftr_env_do_sampling ()) {
        vftr_write_to_vfd(func_exit_time, prof_current, func->id, SID_EXIT);
#ifdef _MPI
        int mpi_isinit;
        PMPI_Initialized(&mpi_isinit);
        if (mpi_isinit) {
           int mpi_isfinal;
           PMPI_Finalized(&mpi_isfinal);
           if (!mpi_isfinal) {
              vftr_clear_completed_requests();
           }
        }
#endif
    }

    /* Maintain profile info */

    prof_current->cycles += cycles0;
    prof_current->time_excl += func_exit_time;
    vftr_prog_cycles += cycles0;
    if (func->return_to) {
        prof_current->cycles -= vftr_prof_data.cycles;
        prof_current->time_excl -= vftr_prof_data.time_excl;
        vftr_prog_cycles -= vftr_prof_data.cycles;
    }

    wall_time = (vftr_get_runtime_usec() - vftr_overhead_usec) * 1.0e-6;

    if (func->profile_this)  {
        vftr_print_stack_at_runtime (func, false, time_to_sample);
        vftr_profile_wanted = true;
    }

    if (timer >= vftr_timelimit) {
        fprintf (vftr_log, "vftr_timelimit exceeded - terminating execution\n");
	kill (getpid(), SIGTERM);
    }

    /* Sort profile if it is time */
    
    if (wall_time >= vftr_sorttime)  {
        int i;
        double tsum = 0.;
        double scale = 100. / (double)vftr_prog_cycles;

        qsort ((void*)vftr_func_table, (size_t)vftr_stackscount, sizeof (function_t *), vftr_get_profile_compare_function());

        /* Set function detail flags while sum(time) < max */
        for (i = 0; i < vftr_stackscount; i++) {
            function_t *f = vftr_func_table[i];
            tsum += (double)f->prof_current.cycles;
	    double cutoff = vftr_environment.detail_until_cum_cycles->value;
            if ((tsum * scale) > cutoff) break;
            f->detail = true;
        }
        /* Clear function detail flags for all others */
        for(; i < vftr_stackscount; i++) 
            vftr_func_table[i]->detail = false;

        vftr_sorttime *= vftr_sorttime_growth;
    }

    /* Compensate overhead */
    // The stuff we did here added up cycles. Therefore, we have to reset
    // the global cycle count and time value.
    vftr_prof_data.cycles = vftr_get_cycles() - vftr_initcycles;
    long long overhead_time_end = vftr_get_runtime_usec();
    vftr_prof_data.time_excl = overhead_time_end;
    vftr_overhead_usec += overhead_time_end - overhead_time_start;
    func->overhead += overhead_time_end - overhead_time_start;

    func->open = false;
}

/**********************************************************************/

// These are the actual Cygnus function hooks. 
//
#if defined(__x86_64__) || defined(__ve__)

void __cyg_profile_func_enter (void *func, void *caller) {
    vftr_function_entry (NULL, func, false);
}

void __cyg_profile_func_exit (void *func, void *caller) {
    vftr_function_exit ();
}

#elif defined(__ia64__)

// The argument func is a pointer to a pointer instead of a pointer.

void __cyg_profile_func_enter (void **func, void *caller) {
    vftr_function_entry (NULL, *func, false);
}

void __cyg_profile_func_exit (void **func, void *caller) {
    vftr_function_exit ();
}
#endif

/**********************************************************************/
