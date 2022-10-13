#include <stdlib.h>

#include <string.h>

#include "sort_utils.h"
#include "sorting.h"
#include "collated_stack_types.h"
#include "profiling_types.h"
#include "environment_types.h"

collated_stack_t **vftr_sort_collated_stacks_for_prof(environment_t environment,
                                                      collated_stacktree_t stacktree) {
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
         collated_stack_t *stack = stacktree.stacks+istack;
         stackvals[istack] += stack->profile.callprof.time_excl_nsec;
      }
   } else if (!strcmp(env_val, "TIME_INCL")) {
      for (int istack=0; istack<nstacks; istack++) {
         collated_stack_t *stack = stacktree.stacks+istack;
         stackvals[istack] += stack->profile.callprof.time_nsec;
      }
   } else if (!strcmp(env_val, "CALLS")) {
      for (int istack=0; istack<nstacks; istack++) {
         collated_stack_t *stack = stacktree.stacks+istack;
         stackvals[istack] += stack->profile.callprof.calls;
      }
   } else if (!strcmp(env_val, "OVERHEAD")) {
      for (int istack=0; istack<nstacks; istack++) {
         collated_stack_t *stack = stacktree.stacks+istack;
         stackvals[istack] += stack->profile.callprof.overhead_nsec;
      }
   } else {
      // if (!strcmp(env_val, "NONE"))
      // if (!strcmp(env_val, "STACK_ID"))
      for (int istack=0; istack<nstacks; istack++) {
         collated_stack_t *stack = stacktree.stacks+istack;
         stackvals[istack] += stack->gid;
      }
      ascending = true;
   }

   // sorting and saving the permutation
   vftr_sort_perm_longlong(nstacks, stackvals, &perm, ascending);
   free(stackvals);

   // create the stackpointer list
   collated_stack_t **stackptrs =
      (collated_stack_t**) malloc(nstacks*sizeof(collated_stack_t*));
   for (int istack=0; istack<nstacks; istack++) {
      stackptrs[istack] = stacktree.stacks+istack;
   }

   // apply the permutation to the stackpointer list,
   // so the stacks are sorted in the same way the slected value is.
   vftr_apply_perm_collated_stackptr(nstacks, stackptrs, perm);
   free(perm);

   return stackptrs;
}

#ifdef _MPI
void vftr_sort_collated_stacks_for_mpiprof(environment_t environment,
                                           int nselected_stacks,
                                           collated_stack_t **selected_stacks) {
   // Depending on the environment variable create a list of that value
   // (summing over the thread individual profiles)
   // sort it and store the permutation to sort a pointerlist pointing
   // to the stacks themselves
   int *perm = NULL;
   long long *stackvals = (long long*) malloc(nselected_stacks*sizeof(long long));
   for (int istack=0; istack<nselected_stacks; istack++) {
      stackvals[istack] = 0ll;
   }
   char *env_val = environment.sort_mpi_table.value.string_val;
   bool ascending = false;

   if (!strcmp(env_val, "MESSAGES")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         collated_stack_t *stack = selected_stacks[istack];
         stackvals[istack] = stack->profile.mpiprof.nsendmessages;
         stackvals[istack] += stack->profile.mpiprof.nrecvmessages;
      }
   } else if (!strcmp(env_val, "SEND_SIZE")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         collated_stack_t *stack = selected_stacks[istack];
         long long avg_send_bytes = stack->profile.mpiprof.send_bytes /
                                    stack->profile.mpiprof.nsendmessages;
         stackvals[istack] = avg_send_bytes;
      }
   } else if (!strcmp(env_val, "RECV_SIZE")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         collated_stack_t *stack = selected_stacks[istack];
         long long avg_recv_bytes = stack->profile.mpiprof.recv_bytes /
                                    stack->profile.mpiprof.nrecvmessages;
         stackvals[istack] = avg_recv_bytes;
      }
   } else if (!strcmp(env_val, "SEND_BW")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         collated_stack_t *stack = selected_stacks[istack];
         long long avg_send_bw = stack->profile.mpiprof.acc_send_bw /
                                 stack->profile.mpiprof.nsendmessages;
         stackvals[istack] = avg_send_bw;
      }
   } else if (!strcmp(env_val, "RECV_BW")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         collated_stack_t *stack = selected_stacks[istack];
         long long avg_recv_bw = stack->profile.mpiprof.acc_recv_bw /
                                 stack->profile.mpiprof.nrecvmessages;
         stackvals[istack] = avg_recv_bw;
      }
   } else if (!strcmp(env_val, "COMM_TIME")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         collated_stack_t *stack = selected_stacks[istack];
         long long avg_comm_time = stack->profile.mpiprof.total_time_nsec / 
            (stack->profile.mpiprof.nsendmessages + stack->profile.mpiprof.nrecvmessages);
         stackvals[istack] = avg_comm_time;
      }
   } else {
      // if (!strcmp(env_val, "NONE"))
      // if (!strcmp(env_val, "STACK_ID"))
      for (int istack=0; istack<nselected_stacks; istack++) {
         collated_stack_t *stack = selected_stacks[istack];
         stackvals[istack] += stack->gid;
      }
      ascending = true;
   }

   // sorting and saving the permutation
   vftr_sort_perm_longlong(nselected_stacks, stackvals, &perm, ascending);
   free(stackvals);

   // apply the permutation to the stackpointer list,
   // so the stacks are sorted in the same way the slected value is.
   vftr_apply_perm_collated_stackptr(nselected_stacks, selected_stacks, perm);
   free(perm);
}
#endif
