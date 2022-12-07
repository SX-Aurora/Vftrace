#include <string.h>
#include <papi.h>

#include "vftrace_state.h"

#include "papiprofiling_types.h"
#include "papi_calculator.h"

papiprofile_t vftr_new_papiprofiling () {
   papiprofile_t prof;
   int n_counters = PAPI_num_events (vftrace.papi_state.eventset);
   prof.counters_incl = (long long*)malloc (n_counters * sizeof(long long));
   memset (prof.counters_incl, 0, n_counters * sizeof(long long));
   prof.counters_excl = (long long*)malloc (n_counters * sizeof(long long));
   memset (prof.counters_excl, 0, n_counters * sizeof(long long));
   return prof;
}

long long *vftr_get_papi_counters () {
  int n = PAPI_num_events (vftrace.papi_state.eventset);
  long long *counters = (long long *)malloc (n * sizeof(long long));
  int retval = PAPI_read (vftrace.papi_state.eventset, counters);
  return counters;
}

void vftr_accumulate_papiprofiling (papiprofile_t *prof, long long *counters, bool invert_sign) {
   int n = PAPI_num_events (vftrace.papi_state.eventset);
   for (int i = 0; i < n; i++) {
      if (invert_sign) {
         prof->counters_incl[i] -= counters[i]; 
      } else {
         prof->counters_incl[i] += counters[i]; 
      }
   }
}

void vftr_update_stacks_exclusive_counters (stacktree_t *stacktree_ptr) {
   int n_counters = vftrace.papi_state.n_counters;
   int nstacks = stacktree_ptr->nstacks;
   vftr_stack_t *stacks = stacktree_ptr->stacks;
   for (int istack = 1; istack < nstacks; istack++) {
      vftr_stack_t *this_stack = stacks + istack;
      for (int iprof = 0; iprof < this_stack->profiling.nprofiles; iprof++) {
         profile_t *this_prof = this_stack->profiling.profiles + iprof;
         for (int e = 0; e < n_counters; e++) {
            this_prof->papiprof.counters_excl[e] = this_prof->papiprof.counters_incl[e];
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
                  this_prof->papiprof.counters_excl[e] -= calleeprof->papiprof.counters_incl[e];
               }
            }
         }
      }
   }
}

void vftr_papiprofiling_free (papiprofile_t *prof_ptr) {
   if (prof_ptr->counters_incl != NULL) free (prof_ptr->counters_incl);
   if (prof_ptr->counters_excl != NULL) free (prof_ptr->counters_excl);
   prof_ptr->counters_incl = NULL;
   prof_ptr->counters_excl = NULL;
}
