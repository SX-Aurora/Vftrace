#include <string.h>

#include "vftrace_state.h"

#include "hwprofiling_types.h"
#include "hwprof_dummy.h"
#ifdef _PAPI_AVAIL
#include "hwprof_papi.h"
#endif
#ifdef _ON_VE
#include "hwprof_ve.h"
#endif 
#include "hwprof_likwid.h"
#include "calculator.h"

hwprofile_t vftr_new_hwprofiling () {
   hwprofile_t prof;
   int n_counters = vftrace.hwprof_state.n_counters;
   int n_observables = vftrace.config.hwprof.observables.obs_name.n_elements;
   prof.counters_incl = (long long*)malloc (n_counters * sizeof(long long));
   memset (prof.counters_incl, 0, n_counters * sizeof(long long));
   prof.counters_excl = (long long*)malloc (n_counters * sizeof(long long));
   memset (prof.counters_excl, 0, n_counters * sizeof(long long));
   prof.observables = (double*)malloc (n_observables * sizeof(double));
   memset (prof.observables, 0, n_observables * sizeof(double));
   return prof;
}

long long *vftr_get_hw_counters () {
   switch (vftrace.hwprof_state.hwc_type) {
      case HWC_DUMMY:
         return vftr_get_dummy_counters(); 
#ifdef _PAPI_AVAIL
      case HWC_PAPI:
         return vftr_get_papi_counters();
#endif
      case HWC_LIKWID:
         return vftr_get_likwid_counters();
#ifdef _ON_VE
      case HWC_VE:
         return vftr_get_active_ve_counters();
#endif
      default:
         //TBD
         return NULL;
   }
}

void vftr_hwprof_adapt_units (hwprof_state_t state, double *value) {
#ifdef _LIKWID_AVAIL
   *value *= state.likwid.energyUnit;
#endif
}

void vftr_accumulate_hwprofiling (hwprofile_t *prof, long long *counters, bool invert_sign) {
   int n = vftrace.hwprof_state.n_counters;
   for (int i = 0; i < n; i++) {
      if (invert_sign) {
         prof->counters_incl[i] -= counters[i]; 
      } else {
         prof->counters_incl[i] += counters[i]; 
      }
   }
}

void vftr_update_stacks_exclusive_counters (stacktree_t *stacktree_ptr) {
   int n_counters = vftrace.hwprof_state.n_counters;
   int nstacks = stacktree_ptr->nstacks;
   vftr_stack_t *stacks = stacktree_ptr->stacks;
   for (int istack = 1; istack < nstacks; istack++) {
      vftr_stack_t *this_stack = stacks + istack;
      for (int iprof = 0; iprof < this_stack->profiling.nprofiles; iprof++) {
         profile_t *this_prof = this_stack->profiling.profiles + iprof;
         for (int e = 0; e < n_counters; e++) {
            this_prof->hwprof.counters_excl[e] = this_prof->hwprof.counters_incl[e];
         } 
         
         for (int icallee = 0; icallee < this_stack->ncallees; icallee++) {
            int calleeID = this_stack->callees[icallee];
            vftr_stack_t *calleestack = stacks + calleeID;    
            int calleeprofID = -1;
            for (int jprof = 0; jprof < calleestack->profiling.nprofiles; jprof++) {
                profile_t *calleeprof = calleestack->profiling.profiles + jprof;
                if (this_prof->threadID == calleeprof->threadID) {
                   calleeprofID = jprof;
                   break;
                }
            }
            if (calleeprofID >= 0) {
               profile_t *calleeprof = calleestack->profiling.profiles + calleeprofID;
               for (int e = 0; e < n_counters; e++) {
                  this_prof->hwprof.counters_excl[e] -= calleeprof->hwprof.counters_incl[e];
               }
            }
         }
      }
   }
}

void vftr_update_stacks_hw_observables (stacktree_t *stacktree_ptr) {
   vftr_calculator_t *calc = &(vftrace.hwprof_state.calculator);
   int nstacks = stacktree_ptr->nstacks;
   vftr_stack_t *stacks = stacktree_ptr->stacks;
   for (int istack = 1; istack < nstacks; istack++) {
      vftr_stack_t *this_stack = stacks + istack;
      for (int iprof = 0; iprof < this_stack->profiling.nprofiles; iprof++) {
         profile_t *this_prof = this_stack->profiling.profiles + iprof;
         callprofile_t callprof = this_prof->callprof;
         hwprofile_t *hwprof = &(this_prof->hwprof);
         vftr_set_calculator_counters (calc, hwprof->counters_excl);
         vftr_set_calculator_builtin (calc, PCALC_T, (double)callprof.time_excl_nsec * 1e-9);
         vftr_set_calculator_builtin (calc, PCALC_CALLS, callprof.calls);
         for (int i = 0; i < calc->n_observables; i++) {
            hwprof->observables[i] = vftr_calculator_evaluate (*calc, i);
            vftr_hwprof_adapt_units (vftrace.hwprof_state, &hwprof->observables[i]);
         }
      }
   }
}

void vftr_hwprofiling_free (hwprofile_t *prof_ptr) {
   if (prof_ptr->counters_incl != NULL) free (prof_ptr->counters_incl);
   if (prof_ptr->counters_excl != NULL) free (prof_ptr->counters_excl);
   if (prof_ptr->observables != NULL) free (prof_ptr->observables);
   prof_ptr->counters_incl = NULL;
   prof_ptr->counters_excl = NULL;
   prof_ptr->observables = NULL;
}
