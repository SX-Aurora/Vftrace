#include <string.h>

#include "status_utils.h"

#include "vftrace_state.h"
#include "collated_stack_types.h"
#include "papiprofiling_types.h"

void vftr_collate_papiprofiles_root_self (collated_stacktree_t *collstacktree_ptr,
			                  stacktree_t *stacktree_ptr) {
   for (int istack = 0; istack < stacktree_ptr->nstacks; istack++) {
      vftr_stack_t *stack = stacktree_ptr->stacks + istack;
      int i_collstack = stack->gid;
      collated_stack_t *collstack = collstacktree_ptr->stacks + i_collstack;

      papiprofile_t copy_papiprof = stack->profiling.profiles[0].papiprof;
      papiprofile_t *collpapiprof = &(collstack->profile.papiprof);
     
      int n_events = PAPI_num_events (vftrace.papi_state.eventset);
      collpapiprof->counters = (long long*)malloc (n_events * sizeof(long long));
      memcpy (collpapiprof->counters, copy_papiprof.counters, n_events * sizeof(long long));
      //for (int i = 0; i < n_events; i++) {
      //   collpapiprof->counters[i] = copy_papiprof.counters[i];
      //   //collpapiprof->counters[i] = 0;
      //}
   }
}

#ifdef _MPI
static void vftr_collate_papiprofiles_on_root (collated_stacktree_t *collstacktree_ptr,
                                               stacktree_t *stacktree_ptr,
					       int myrank, int nranks, int *nremote_profiles) {

   int num_counters = PAPI_num_events (vftrace.papi_state.eventset);
   if (myrank > 0) {
      int nprofiles = stacktree_ptr->nstacks;
      int *gids = (int*) malloc(nprofiles * sizeof(int));
      long long *sendbuf = (long long*)malloc(nprofiles * num_counters * sizeof(long long));

      for (int istack = 0; istack < nprofiles; istack++) {
         vftr_stack_t *mystack = stacktree_ptr->stacks + istack;
         gids[istack] = mystack->gid;
         profile_t *myprof = mystack->profiling.profiles;
         for (int e = 0; e < num_counters; e++) { 
            sendbuf[istack * num_counters + e] = myprof->papiprof.counters[e];
         }
      }
      PMPI_Send (gids, nprofiles, MPI_INT, 0, myrank, MPI_COMM_WORLD);
      PMPI_Send (sendbuf, nprofiles * num_counters, MPI_LONG_LONG, 0, myrank, MPI_COMM_WORLD);
      free(gids);
      free(sendbuf);
   } else {
      int maxprofiles = 0;
      for (int irank = 1; irank < nranks; irank++) {
         maxprofiles = nremote_profiles[irank] > maxprofiles ? nremote_profiles[irank] : maxprofiles;
      }

      for (int irank = 1; irank < nranks; irank++) {
         int nprofiles = nremote_profiles[irank];
         int *gids = (int*)malloc(nprofiles * sizeof(int));
         long long *recvbuf = (long long*)malloc(nprofiles * num_counters * sizeof(long long));
         MPI_Status status;
         PMPI_Recv (gids, nprofiles, MPI_INT, irank, irank, MPI_COMM_WORLD, &status);
         PMPI_Recv (recvbuf, nprofiles * num_counters, MPI_LONG_LONG, irank, irank, MPI_COMM_WORLD, &status);
         for (int iprof = 0; iprof < nprofiles; iprof++) {
            int gid = gids[iprof]; 
            collated_stack_t *collstack = collstacktree_ptr->stacks + gid;
            papiprofile_t *collpapiprof = &(collstack->profile.papiprof);

            for (int e = 0; e < num_counters; e++) {
               collpapiprof->counters[e] += recvbuf[iprof * num_counters + e];
            }
         }
         free(gids);
         free(recvbuf); 
      }
   }
}
#endif

void vftr_collate_papiprofiles (collated_stacktree_t *collstacktree_ptr,
			        stacktree_t *stacktree_ptr,
				int myrank, int nranks, int *nremote_profiles) {
   vftr_collate_papiprofiles_root_self(collstacktree_ptr, stacktree_ptr);
#ifdef _MPI
   int mpi_initialized;
   PMPI_Initialized(&mpi_initialized);
   if (mpi_initialized) {
      vftr_collate_papiprofiles_on_root (collstacktree_ptr, stacktree_ptr, myrank, nranks, nremote_profiles);
   }
#endif
}
