#include <stdlib.h>

#include <string.h>

#include "sort_utils.h"
#include "sorting.h"
#include "collated_stack_types.h"
#include "profiling_types.h"
#include "configuration_types.h"

collated_stack_t **vftr_sort_collated_stacks_for_prof(config_t config,
                                                      collated_stacktree_t stacktree) {
   int nstacks = stacktree.nstacks;
   // Depending on the configuration variable create a list of that value
   // (summing over the thread individual profiles)
   // sort it and store the permutation to sort a pointerlist pointing
   // to the stacks themselves
   int *perm = NULL;
   long long *stackvals = (long long*) malloc(nstacks*sizeof(long long));
   for (int istack=0; istack<nstacks; istack++) {
      stackvals[istack] = 0ll;
   }
   char *column = config.profile_table.sort_table.column.value;
   bool ascending = config.profile_table.sort_table.ascending.value;
   if (!strcmp(column, "time_excl")) {
      for (int istack=0; istack<nstacks; istack++) {
         collated_stack_t *stack = stacktree.stacks+istack;
         stackvals[istack] += stack->profile.callprof.time_excl_nsec;
      }
   } else if (!strcmp(column, "time_incl")) {
      for (int istack=0; istack<nstacks; istack++) {
         collated_stack_t *stack = stacktree.stacks+istack;
         stackvals[istack] += stack->profile.callprof.time_nsec;
      }
   } else if (!strcmp(column, "calls")) {
      for (int istack=0; istack<nstacks; istack++) {
         collated_stack_t *stack = stacktree.stacks+istack;
         stackvals[istack] += stack->profile.callprof.calls;
      }
   } else if (!strcmp(column, "overhead")) {
      for (int istack=0; istack<nstacks; istack++) {
         collated_stack_t *stack = stacktree.stacks+istack;
         stackvals[istack] += stack->profile.callprof.overhead_nsec;
      }
   } else {
      // if (!strcmp(column, "none"))
      // if (!strcmp(column, "stack_id"))
      if (stacktree.namegrouped) {
         for (int istack=0; istack<nstacks; istack++) {
            collated_stack_t *stack = stacktree.stacks+istack;
            stackvals[istack] += stack->gid_list.gids[0];
         }
      } else {
         for (int istack=0; istack<nstacks; istack++) {
            collated_stack_t *stack = stacktree.stacks+istack;
            stackvals[istack] += stack->gid;
         }
      }
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
void vftr_sort_collated_stacks_for_mpiprof(config_t config,
                                           int nselected_stacks,
                                           collated_stack_t **selected_stacks) {
   // Depending on the configuration variable create a list of that value
   // (summing over the thread individual profiles)
   // sort it and store the permutation to sort a pointerlist pointing
   // to the stacks themselves
   int *perm = NULL;
   long long *stackvals = (long long*) malloc(nselected_stacks*sizeof(long long));
   for (int istack=0; istack<nselected_stacks; istack++) {
      stackvals[istack] = 0ll;
   }
   char *column = config.mpi.sort_table.column.value;
   bool ascending = config.mpi.sort_table.ascending.value;

   if (!strcmp(column, "messages")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         collated_stack_t *stack = selected_stacks[istack];
         stackvals[istack] = stack->profile.mpiprof.nsendmessages;
         stackvals[istack] += stack->profile.mpiprof.nrecvmessages;
      }
   } else if (!strcmp(column, "send_size")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         collated_stack_t *stack = selected_stacks[istack];
         long long avg_send_bytes = stack->profile.mpiprof.send_bytes /
                                    stack->profile.mpiprof.nsendmessages;
         stackvals[istack] = avg_send_bytes;
      }
   } else if (!strcmp(column, "recv_size")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         collated_stack_t *stack = selected_stacks[istack];
         long long avg_recv_bytes = stack->profile.mpiprof.recv_bytes /
                                    stack->profile.mpiprof.nrecvmessages;
         stackvals[istack] = avg_recv_bytes;
      }
   } else if (!strcmp(column, "send_bw")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         collated_stack_t *stack = selected_stacks[istack];
         long long avg_send_bw = stack->profile.mpiprof.acc_send_bw /
                                 stack->profile.mpiprof.nsendmessages;
         stackvals[istack] = avg_send_bw;
      }
   } else if (!strcmp(column, "recv_bw")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         collated_stack_t *stack = selected_stacks[istack];
         long long avg_recv_bw = stack->profile.mpiprof.acc_recv_bw /
                                 stack->profile.mpiprof.nrecvmessages;
         stackvals[istack] = avg_recv_bw;
      }
   } else if (!strcmp(column, "comm_time")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         collated_stack_t *stack = selected_stacks[istack];
         long long avg_comm_time = stack->profile.mpiprof.total_time_nsec / 
            (stack->profile.mpiprof.nsendmessages + stack->profile.mpiprof.nrecvmessages);
         stackvals[istack] = avg_comm_time;
      }
   } else {
      // if (!strcmp(column, "none"))
      // if (!strcmp(column, "stack_id"))
      for (int istack=0; istack<nselected_stacks; istack++) {
         collated_stack_t *stack = selected_stacks[istack];
         stackvals[istack] += stack->gid;
      }
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

#ifdef _CUDA
collated_stack_t **vftr_sort_collated_stacks_for_cuda (config_t config, collated_stacktree_t stacktree) {
     int nstacks = stacktree.nstacks;

     char *column = config.cuda.sort_table.column.value;
     bool ascending = config.cuda.sort_table.ascending.value;
     int *perm = NULL;

     if (!strcmp(column, "time")) {
        float *stackvals = (float*)malloc(nstacks * sizeof(float));
        for (int istack = 0; istack < nstacks; istack++) {
           collated_stack_t *stack = stacktree.stacks + istack;
           stackvals[istack] = stack->profile.cudaprof.t_ms;
        }
        vftr_sort_perm_float(nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else if (!strcmp(column, "memcpy")) {
        long long *stackvals = (long long*)malloc(nstacks * sizeof(long long));
        for (int istack = 0; istack < nstacks; istack++) {
           collated_stack_t *stack = stacktree.stacks + istack;
           collated_profile_t prof = stack->profile;
           stackvals[istack] = (long long)(prof.cudaprof.memcpy_bytes[0] + prof.cudaprof.memcpy_bytes[1]);
        }
        vftr_sort_perm_longlong(nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else if (!strcmp(column, "cbid")) {
        int *stackvals = (int*)malloc(nstacks * sizeof(int));
        for (int istack = 0; istack < nstacks; istack++) {
           collated_stack_t *stack = stacktree.stacks + istack;
           stackvals[istack] = stack->profile.cudaprof.cbid;
        }
        vftr_sort_perm_int(nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else if (!strcmp(column, "calls")) {
        int *stackvals = (int*)malloc(nstacks * sizeof(int));
        for (int istack = 0; istack < nstacks; istack++) {
           collated_stack_t *stack = stacktree.stacks + istack;
           stackvals[istack] = stack->profile.cudaprof.n_calls[0] + stack->profile.cudaprof.n_calls[1];
        }
        vftr_sort_perm_int(nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else {
        // if (!strcmp(column, "none"))
        int *stackvals = (int*)malloc(nstacks * sizeof(int));
        for (int istack = 0; istack < nstacks; istack++) {
           collated_stack_t *stack = stacktree.stacks + istack;
           stackvals[istack] = stack->gid;
        }
        vftr_sort_perm_int(nstacks, stackvals, &perm, ascending);
        free(stackvals);
     }

     collated_stack_t **stackptrs = (collated_stack_t**) malloc(nstacks*sizeof(collated_stack_t*));
     for (int istack = 0; istack < nstacks; istack++) {
        stackptrs[istack] = stacktree.stacks + istack;
     }

     vftr_apply_perm_collated_stackptr (nstacks, stackptrs, perm);
     free(perm);
     return stackptrs; 
  }
#endif

#ifdef _ACCPROF
collated_stack_t **vftr_sort_collated_stacks_for_accprof (config_t config, collated_stacktree_t stacktree) {
     int nstacks = stacktree.nstacks;
     char *column = config.accprof.sort_table.column.value;
     bool ascending = config.accprof.sort_table.ascending.value;
     int *perm = NULL;

     if (!strcmp(column, "time")) {
        long long *stackvals = (long long *)malloc(nstacks * sizeof(long long)); 
        for (int istack = 0; istack < nstacks; istack++) {
           collated_stack_t *stack = stacktree.stacks + istack;
           stackvals[istack] = stack->profile.callprof.time_excl_nsec;
        }
        vftr_sort_perm_longlong (nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else if (!strcmp(column, "memcpy")) {
        long long *stackvals = (long long *)malloc(nstacks * sizeof(long long)); 
        for (int istack = 0; istack < nstacks; istack++) {
           collated_stack_t *stack = stacktree.stacks + istack;
           stackvals[istack] = stack->profile.accprof.copied_bytes;
        }
        vftr_sort_perm_longlong (nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else if (!strcmp(column, "evtype")) {
        int *stackvals = (int *)malloc(nstacks * sizeof(int));
        for (int istack = 0; istack < nstacks; istack++) {
           collated_stack_t *stack = stacktree.stacks + istack;
           stackvals[istack] = (int)stack->profile.accprof.event_type;
        }
        vftr_sort_perm_int (nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else if (!strcmp(column, "calls")) {
        int *stackvals = (int *)malloc(nstacks * sizeof(int)); 
        for (int istack = 0; istack < nstacks; istack++) {
           collated_stack_t *stack = stacktree.stacks + istack;
           stackvals[istack] = stack->profile.callprof.calls;
        }
        vftr_sort_perm_int (nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else {
	// sort = none
	int *stackvals = (int*)malloc(nstacks * sizeof(int));
        for (int istack = 0; istack < nstacks; istack++) {
           collated_stack_t *stack = stacktree.stacks + istack;
           stackvals[istack] = stack->gid;
        }
        vftr_sort_perm_int(nstacks, stackvals, &perm, ascending);
        free(stackvals);
    }

    collated_stack_t **stackptrs = (collated_stack_t**) malloc(nstacks*sizeof(collated_stack_t*));
    for (int istack = 0; istack < nstacks; istack++) {
       stackptrs[istack] = stacktree.stacks + istack;
    }

    vftr_apply_perm_collated_stackptr (nstacks, stackptrs, perm);
    free(perm);
    return stackptrs; 
  }

#endif

collated_stack_t **vftr_sort_collated_stacks_hwprof_obs (config_t config, collated_stacktree_t stacktree) {
     int nstacks = stacktree.nstacks;
     int sort_column = config.hwprof.sort_by_column.value;
     int *perm = NULL;
     double *observables   = (double*)malloc(nstacks * sizeof(long long));
     for (int istack = 0; istack < nstacks; istack++) {
        collated_stack_t *stack = stacktree.stacks + istack;
        hwprofile_t hwprof = stack->profile.hwprof;
        observables[istack] = hwprof.observables[sort_column];
     }

     vftr_sort_perm_double (nstacks, observables, &perm, false);
     free(observables);

     collated_stack_t **stackptrs = (collated_stack_t**) malloc(nstacks*sizeof(collated_stack_t*));
     for (int istack = 0; istack < nstacks; istack++) {
        stackptrs[istack] = stacktree.stacks + istack;
     }

     vftr_apply_perm_collated_stackptr (nstacks, stackptrs, perm);
     free(perm);
     return stackptrs; 
}

collated_stack_t **vftr_sort_collated_stacks_tmax (config_t config, collated_stacktree_t stacktree) {
    int nstacks = stacktree.nstacks;
    long long *t_max = (long long*)malloc(nstacks * sizeof(long long));
    for (int istack = 0; istack < nstacks; istack++) {
       collated_stack_t *stack = stacktree.stacks + istack;
       collated_callprofile_t callprof = stack->profile.callprof;
       t_max[istack] = callprof.max_time_nsec; 
    }

    int *perm = NULL;
    vftr_sort_perm_longlong (nstacks, t_max, &perm, false);
    free(t_max);

    collated_stack_t **stackptrs = (collated_stack_t**) malloc(nstacks*sizeof(collated_stack_t*));
    for (int istack = 0; istack < nstacks; istack++) {
        stackptrs[istack] = stacktree.stacks + istack;
    }

    vftr_apply_perm_collated_stackptr (nstacks, stackptrs, perm);
    free(perm);
    return stackptrs; 
}
