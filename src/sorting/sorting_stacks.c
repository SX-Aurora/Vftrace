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
            stackvals[istack] += prof->callprof.time_excl_nsec;
         }
      }
   } else if (!strcmp(env_val, "TIME_INCL")) {
      for (int istack=0; istack<nstacks; istack++) {
         stack_t *stack = stacktree.stacks+istack;
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            stackvals[istack] += prof->callprof.time_nsec;
         }
      }
   } else if (!strcmp(env_val, "CALLS")) {
      for (int istack=0; istack<nstacks; istack++) {
         stack_t *stack = stacktree.stacks+istack;
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            stackvals[istack] += prof->callprof.calls;
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
            stackvals[istack] += prof->callprof.overhead_nsec;
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

#ifdef _MPI
void vftr_sort_stacks_for_mpiprof(environment_t environment,
                                  int nselected_stacks,
                                  stack_t **selected_stacks) {
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
         stack_t *stack = selected_stacks[istack];
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            stackvals[istack] += prof->mpiprof.nsendmessages;
            stackvals[istack] += prof->mpiprof.nrecvmessages;
         }
      }
   } else if (!strcmp(env_val, "SEND_SIZE")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         stack_t *stack = selected_stacks[istack];
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            long long avg_send_bytes = prof->mpiprof.send_bytes /
                                       prof->mpiprof.nsendmessages;
            stackvals[istack] += avg_send_bytes;
         }
      }
   } else if (!strcmp(env_val, "RECV_SIZE")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         stack_t *stack = selected_stacks[istack];
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            long long avg_recv_bytes = prof->mpiprof.recv_bytes /
                                       prof->mpiprof.nrecvmessages;
            stackvals[istack] += avg_recv_bytes;
         }
      }
   } else if (!strcmp(env_val, "SEND_BW")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         stack_t *stack = selected_stacks[istack];
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            long long avg_recv_bw = prof->mpiprof.acc_send_bw /
                                    prof->mpiprof.nsendmessages;
            stackvals[istack] = avg_recv_bw;
         }
      }
   } else if (!strcmp(env_val, "RECV_BW")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         stack_t *stack = selected_stacks[istack];
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            long long avg_recv_bw = prof->mpiprof.acc_recv_bw /
                                    prof->mpiprof.nrecvmessages;
            stackvals[istack] = avg_recv_bw;
         }
      }
   } else if (!strcmp(env_val, "COMM_TIME")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         stack_t *stack = selected_stacks[istack];
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            long long avg_comm_time = prof->mpiprof.total_time_nsec /
               (prof->mpiprof.nsendmessages + prof->mpiprof.nrecvmessages);
            stackvals[istack] = avg_comm_time;
         }
      }
   } else if (!strcmp(env_val, "STACK_ID")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         stack_t *stack = selected_stacks[istack];
         stackvals[istack] = stack->gid;
      }
      ascending = true;
   } else {
      // if (!strcmp(env_val, "NONE"))
      for (int istack=0; istack<nselected_stacks; istack++) {
         stack_t *stack = selected_stacks[istack];
         stackvals[istack] += stack->lid;
      }
      ascending = true;
   }

   // sorting and saving the permutation
   vftr_sort_perm_longlong(nselected_stacks, stackvals, &perm, ascending);
   free(stackvals);

   // apply the permutation to the stackpointer list,
   // so the stacks are sorted in the same way the slected value is.
   vftr_apply_perm_stackptr(nselected_stacks, selected_stacks, perm);
   free(perm);
}
#endif

#ifdef _CUDA
   stack_t **vftr_sort_stacks_for_cuda (environment_t environment, stacktree_t stacktree) {
     int nstacks = stacktree.nstacks;

     char *env_val = environment.sort_cuda_table.value.string_val;
     int *perm = NULL;

     if (!strcmp(env_val, "TIME")) {
        float *stackvals = (float*)malloc(nstacks * sizeof(float));
        for (int istack = 0; istack < nstacks; istack++) {
           stack_t *stack = stacktree.stacks + istack;
           profile_t *prof = stack->profiling.profiles;
           stackvals[istack] = prof->cudaprof.t_ms;
        }
        vftr_sort_perm_float(nstacks, stackvals, &perm, false);
        free(stackvals);
     } else if (!strcmp(env_val, "MEMCPY")) {
        long long  *stackvals = (long long*)malloc(nstacks * sizeof(long long));
        for (int istack = 0; istack < nstacks; istack++) {
           stack_t *stack = stacktree.stacks + istack;
           profile_t *prof = stack->profiling.profiles;
           stackvals[istack] = (long long)(prof->cudaprof.memcpy_bytes[0] + prof->cudaprof.memcpy_bytes[1]);
        }
        vftr_sort_perm_longlong(nstacks, stackvals, &perm, false);
        free(stackvals);
     } else if (!strcmp(env_val, "CBID")) {
        int *stackvals = (int*)malloc(nstacks * sizeof(int));
        for (int istack = 0; istack < nstacks; istack++) {
           stack_t *stack = stacktree.stacks + istack;
           profile_t *prof = stack->profiling.profiles;
           stackvals[istack] = prof->cudaprof.cbid;
        }
        // CBIDs are sorted in ascending order
        vftr_sort_perm_int(nstacks, stackvals, &perm, true);
        free(stackvals);
     } else if (!strcmp(env_val, "CALLS")) {
        int *stackvals = (int*)malloc(nstacks * sizeof(int));
        for (int istack = 0; istack < nstacks; istack++) {
           stack_t *stack = stacktree.stacks + istack;
           profile_t *prof = stack->profiling.profiles;
           stackvals[istack] = prof->cudaprof.n_calls;
        }
        vftr_sort_perm_int(nstacks, stackvals, &perm, false);
        free(stackvals);
     } else {
        // if (!strcmp(env_val, "NONE"))
        int *stackvals = (int*)malloc(nstacks * sizeof(int));
        for (int istack = 0; istack < nstacks; istack++) {
           stack_t *stack = stacktree.stacks + istack;
           stackvals[istack] = stack->lid;
        }
        // Stacktrees are created in ascending order. We need to keep this.
        vftr_sort_perm_int(nstacks, stackvals, &perm, true);
        free(stackvals);
     }

     stack_t **stackptrs = (stack_t**) malloc(nstacks*sizeof(stack_t*));
     for (int istack = 0; istack < nstacks; istack++) {
        stackptrs[istack] = stacktree.stacks + istack;
     }

     vftr_apply_perm_stackptr (nstacks, stackptrs, perm);
     free(perm);
     return stackptrs; 
}
#endif
