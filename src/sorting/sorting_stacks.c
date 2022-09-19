#include <stdlib.h>

#include <string.h>

#include "sort_utils.h"
#include "sorting.h"
#include "stack_types.h"
#include "profiling_types.h"
#include "environment_types.h"

stack_t **vftr_sort_stacks_for_prof(environment_t environment,
                                    stacktree_t stacktree) {
   int nstacks = stacktree.nstacks;
   // Depending on the environment variable create a list of that value
   // (summing over the thread individual profiles)
   // sort it and store the permutation to sort a pointerlist pointing
   // to the stacks themselves
   int *perm = NULL;
   long long *stackvals = (long long*) malloc(nstacks*sizeof(long long));
   for (int istack=0; istack<nstacks; istack++) {
      stackvals[istack] = 0ll;
   }
   char *env_val = environment.sort_profile_table.value.string_val;
   bool ascending = false;
   if (!strcmp(env_val, "TIME_EXCL")) {
      for (int istack=0; istack<nstacks; istack++) {
         stack_t *stack = stacktree.stacks+istack;
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            stackvals[istack] += prof->callProf.time_excl_nsec;
         }
      }
   } else if (!strcmp(env_val, "TIME_INCL")) {
      for (int istack=0; istack<nstacks; istack++) {
         stack_t *stack = stacktree.stacks+istack;
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            stackvals[istack] += prof->callProf.time_nsec;
         }
      }
   } else if (!strcmp(env_val, "CALLS")) {
      for (int istack=0; istack<nstacks; istack++) {
         stack_t *stack = stacktree.stacks+istack;
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            stackvals[istack] += prof->callProf.calls;
         }
      }
   } else if (!strcmp(env_val, "STACK_ID")) {
      for (int istack=0; istack<nstacks; istack++) {
         stack_t *stack = stacktree.stacks+istack;
         stackvals[istack] = stack->gid;
      }
      ascending = true;
   } else if (!strcmp(env_val, "OVERHEAD")) {
      for (int istack=0; istack<nstacks; istack++) {
         stack_t *stack = stacktree.stacks+istack;
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            stackvals[istack] += prof->callProf.overhead_nsec;
         }
      }
   } else {
      // if (!strcmp(env_val, "NONE"))
      for (int istack=0; istack<nstacks; istack++) {
         stack_t *stack = stacktree.stacks+istack;
         stackvals[istack] += stack->lid;
      }
      ascending = true;
   }

   // sorting and saving the permutation
   vftr_sort_perm_longlong(nstacks, stackvals, &perm, ascending);
   free(stackvals);

   // create the stackpointer list
   stack_t **stackptrs = (stack_t**) malloc(nstacks*sizeof(stack_t*));
   for (int istack=0; istack<nstacks; istack++) {
      stackptrs[istack] = stacktree.stacks+istack;
   }

   // apply the permutation to the stackpointer list,
   // so the stacks are sorted in the same way the slected value is.
   vftr_apply_perm_stackptr(nstacks, stackptrs, perm);
   free(perm);

   return stackptrs;
}
