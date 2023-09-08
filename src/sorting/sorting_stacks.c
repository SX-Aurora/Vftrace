#include <stdlib.h>

#include <string.h>

#include "sort_utils.h"
#include "sorting.h"
#include "stack_types.h"
#include "profiling_types.h"
#include "configuration_types.h"

vftr_stack_t **vftr_sort_stacks_for_prof(config_t config,
                                    stacktree_t stacktree) {
   int nstacks = stacktree.nstacks;
   // Depending on the config variable create a list of that value
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
         vftr_stack_t *stack = stacktree.stacks+istack;
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            stackvals[istack] += prof->callprof.time_excl_nsec;
         }
      }
   } else if (!strcmp(column, "time_incl")) {
      for (int istack=0; istack<nstacks; istack++) {
         vftr_stack_t *stack = stacktree.stacks+istack;
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            stackvals[istack] += prof->callprof.time_nsec;
         }
      }
   } else if (!strcmp(column, "calls")) {
      for (int istack=0; istack<nstacks; istack++) {
         vftr_stack_t *stack = stacktree.stacks+istack;
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            stackvals[istack] += prof->callprof.calls;
         }
      }
   } else if (!strcmp(column, "stack_id")) {
      for (int istack=0; istack<nstacks; istack++) {
         vftr_stack_t *stack = stacktree.stacks+istack;
         stackvals[istack] = stack->gid;
      }
   } else if (!strcmp(column, "overhead")) {
      for (int istack=0; istack<nstacks; istack++) {
         vftr_stack_t *stack = stacktree.stacks+istack;
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            stackvals[istack] += prof->callprof.overhead_nsec;
         }
      }
   } else {
      // if (!strcmp(column, "none"))
      for (int istack=0; istack<nstacks; istack++) {
         vftr_stack_t *stack = stacktree.stacks+istack;
         stackvals[istack] += stack->lid;
      }
   }

   // sorting and saving the permutation
   vftr_sort_perm_longlong(nstacks, stackvals, &perm, ascending);
   free(stackvals);

   // create the stackpointer list
   vftr_stack_t **stackptrs = (vftr_stack_t**) malloc(nstacks*sizeof(vftr_stack_t*));
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
void vftr_sort_stacks_for_mpiprof(config_t config,
                                  int nselected_stacks,
                                  vftr_stack_t **selected_stacks) {
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
         vftr_stack_t *stack = selected_stacks[istack];
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            stackvals[istack] += prof->mpiprof.nsendmessages;
            stackvals[istack] += prof->mpiprof.nrecvmessages;
         }
      }
   } else if (!strcmp(column, "send_size")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         vftr_stack_t *stack = selected_stacks[istack];
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            long long avg_send_bytes = prof->mpiprof.send_bytes /
                                       prof->mpiprof.nsendmessages;
            stackvals[istack] += avg_send_bytes;
         }
      }
   } else if (!strcmp(column, "recv_size")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         vftr_stack_t *stack = selected_stacks[istack];
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            long long avg_recv_bytes = prof->mpiprof.recv_bytes /
                                       prof->mpiprof.nrecvmessages;
            stackvals[istack] += avg_recv_bytes;
         }
      }
   } else if (!strcmp(column, "send_bw")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         vftr_stack_t *stack = selected_stacks[istack];
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            long long avg_recv_bw = prof->mpiprof.acc_send_bw /
                                    prof->mpiprof.nsendmessages;
            stackvals[istack] = avg_recv_bw;
         }
      }
   } else if (!strcmp(column, "recv_bw")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         vftr_stack_t *stack = selected_stacks[istack];
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            long long avg_recv_bw = prof->mpiprof.acc_recv_bw /
                                    prof->mpiprof.nrecvmessages;
            stackvals[istack] = avg_recv_bw;
         }
      }
   } else if (!strcmp(column, "comm_time")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         vftr_stack_t *stack = selected_stacks[istack];
         int nprofs = stack->profiling.nprofiles;
         for (int iprof=0; iprof<nprofs; iprof++) {
            profile_t *prof = stack->profiling.profiles+iprof;
            long long avg_comm_time = prof->mpiprof.total_time_nsec /
               (prof->mpiprof.nsendmessages + prof->mpiprof.nrecvmessages);
            stackvals[istack] = avg_comm_time;
         }
      }
   } else if (!strcmp(column, "stack_id")) {
      for (int istack=0; istack<nselected_stacks; istack++) {
         vftr_stack_t *stack = selected_stacks[istack];
         stackvals[istack] = stack->gid;
      }
   } else {
      // if (!strcmp(column, "none"))
      for (int istack=0; istack<nselected_stacks; istack++) {
         vftr_stack_t *stack = selected_stacks[istack];
         stackvals[istack] += stack->lid;
      }
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
   vftr_stack_t **vftr_sort_stacks_for_cuda (config_t config, stacktree_t stacktree) {
     int nstacks = stacktree.nstacks;
     char *column = config.cuda.sort_table.column.value;
     bool ascending = config.cuda.sort_table.ascending.value;
     int *perm = NULL;

     if (!strcmp(column, "time")) {
        float *stackvals = (float*)malloc(nstacks * sizeof(float));
        for (int istack = 0; istack < nstacks; istack++) {
           vftr_stack_t *stack = stacktree.stacks + istack;
           profile_t *prof = stack->profiling.profiles;
           stackvals[istack] = prof->cudaprof.t_ms;
        }
        vftr_sort_perm_float(nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else if (!strcmp(column, "memcpy")) {
        long long  *stackvals = (long long*)malloc(nstacks * sizeof(long long));
        for (int istack = 0; istack < nstacks; istack++) {
           vftr_stack_t *stack = stacktree.stacks + istack;
           profile_t *prof = stack->profiling.profiles;
           stackvals[istack] = (long long)(prof->cudaprof.memcpy_bytes[0] + prof->cudaprof.memcpy_bytes[1]);
        }
        vftr_sort_perm_longlong(nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else if (!strcmp(column, "cbid")) {
        int *stackvals = (int*)malloc(nstacks * sizeof(int));
        for (int istack = 0; istack < nstacks; istack++) {
           vftr_stack_t *stack = stacktree.stacks + istack;
           profile_t *prof = stack->profiling.profiles;
           stackvals[istack] = prof->cudaprof.cbid;
        }
        vftr_sort_perm_int(nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else if (!strcmp(column, "calls")) {
        int *stackvals = (int*)malloc(nstacks * sizeof(int));
        for (int istack = 0; istack < nstacks; istack++) {
           vftr_stack_t *stack = stacktree.stacks + istack;
           profile_t *prof = stack->profiling.profiles;
           stackvals[istack] = prof->cudaprof.n_calls[0] + prof->cudaprof.n_calls[1];
        }
        vftr_sort_perm_int(nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else {
        // if (!strcmp(column, "none"))
        int *stackvals = (int*)malloc(nstacks * sizeof(int));
        for (int istack = 0; istack < nstacks; istack++) {
           vftr_stack_t *stack = stacktree.stacks + istack;
           stackvals[istack] = stack->lid;
        }
        vftr_sort_perm_int(nstacks, stackvals, &perm, ascending);
        free(stackvals);
     }

     vftr_stack_t **stackptrs = (vftr_stack_t**) malloc(nstacks*sizeof(vftr_stack_t*));
     for (int istack = 0; istack < nstacks; istack++) {
        stackptrs[istack] = stacktree.stacks + istack;
     }

     vftr_apply_perm_stackptr (nstacks, stackptrs, perm);
     free(perm);
     return stackptrs; 
}
#endif

#ifdef _ACCPROF
   vftr_stack_t **vftr_sort_stacks_for_accprof (config_t config, stacktree_t stacktree) {
     int nstacks = stacktree.nstacks;
     char *column = config.cuda.sort_table.column.value;
     bool ascending = config.cuda.sort_table.ascending.value;
     int *perm = NULL;

     if (!strcmp(column, "time")) {
        long long *stackvals = (long long*)malloc(nstacks * sizeof(long long));
        for (int istack = 0; istack < nstacks; istack++) {
           vftr_stack_t *stack = stacktree.stacks + istack;
           profile_t *prof = stack->profiling.profiles;
           stackvals[istack] = prof->callprof.time_excl_nsec;
        }
        vftr_sort_perm_longlong (nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else if (!strcmp(column, "memcpy")) {
        long long *stackvals = (long long*)malloc(nstacks * sizeof(long long));
        for (int istack = 0; istack < nstacks; istack++) {
           vftr_stack_t *stack = stacktree.stacks + istack;
           profile_t *prof = stack->profiling.profiles;
           stackvals[istack] = prof->accprof.copied_bytes;
        }
        vftr_sort_perm_longlong(nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else if (!strcmp(column, "evtype")) {
        int *stackvals = (int*)malloc(nstacks * sizeof(int));
        for (int istack = 0; istack < nstacks; istack++) {
           vftr_stack_t *stack = stacktree.stacks + istack;
           profile_t *prof = stack->profiling.profiles;
           stackvals[istack] = prof->accprof.event_type;
        }
        vftr_sort_perm_int(nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else if (!strcmp(column, "calls")) {
        int *stackvals = (int*)malloc(nstacks * sizeof(int));
        for (int istack = 0; istack < nstacks; istack++) {
           vftr_stack_t *stack = stacktree.stacks + istack;
           profile_t *prof = stack->profiling.profiles;
           stackvals[istack] = prof->callprof.calls;
        }
        vftr_sort_perm_int(nstacks, stackvals, &perm, ascending);
        free(stackvals);
     } else {
        // if (!strcmp(column, "none"))
        int *stackvals = (int*)malloc(nstacks * sizeof(int));
        for (int istack = 0; istack < nstacks; istack++) {
           vftr_stack_t *stack = stacktree.stacks + istack;
           stackvals[istack] = stack->lid;
        }
        vftr_sort_perm_int(nstacks, stackvals, &perm, ascending);
        free(stackvals);
     }

     vftr_stack_t **stackptrs = (vftr_stack_t**) malloc(nstacks*sizeof(vftr_stack_t*));
     for (int istack = 0; istack < nstacks; istack++) {
        stackptrs[istack] = stacktree.stacks + istack;
     }

     vftr_apply_perm_stackptr (nstacks, stackptrs, perm);
     free(perm);
     return stackptrs; 
}
#endif

  vftr_stack_t **vftr_sort_stacks_hwprof_obs (config_t config, stacktree_t stacktree) {
     int sort_column = config.hwprof.sort_by_column.value;
     int nstacks = stacktree.nstacks; 
     int *perm = NULL;
     double *observables   = (double*)malloc(nstacks * sizeof(long long));
     for (int istack = 0; istack < nstacks; istack++) {
        vftr_stack_t *stack = stacktree.stacks + istack;
        hwprofile_t hwprof = stack->profiling.profiles[0].hwprof;
        observables[istack] = hwprof.observables[sort_column];
     }

     vftr_sort_perm_double(nstacks, observables, &perm, false);
     free(observables);

     vftr_stack_t **stackptrs = (vftr_stack_t**)malloc(nstacks * sizeof(vftr_stack_t*));
     for (int istack = 0; istack < nstacks; istack++) {
        stackptrs[istack] = stacktree.stacks + istack;
     }
     
     vftr_apply_perm_stackptr (nstacks, stackptrs, perm);
     free(perm);
     return stackptrs;
} 
