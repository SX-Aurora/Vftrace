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
     
      int n_events = vftrace.papi_state.n_counters;
      int n_observables = vftrace.config.papi.observables.obs_name.n_elements;
      collpapiprof->counters_incl = (long long*)malloc (n_events * sizeof(long long));
      collpapiprof->counters_excl = (long long*)malloc (n_events * sizeof(long long));
      collpapiprof->observables = (double*)malloc (n_observables * sizeof(double));
      memcpy (collpapiprof->counters_incl, copy_papiprof.counters_incl, n_events * sizeof(long long));
      memcpy (collpapiprof->counters_excl, copy_papiprof.counters_excl, n_events * sizeof(long long));
      memcpy (collpapiprof->observables, copy_papiprof.observables, n_observables * sizeof(double));
   
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

   int num_counters = vftrace.papi_state.n_counters;
   int n_observables = vftrace.config.papi.observables.obs_name.n_elements;
   if (myrank > 0) {
      int nprofiles = stacktree_ptr->nstacks;
      int *gids = (int*) malloc(nprofiles * sizeof(int));
      long long *sendbuf_incl = (long long*)malloc(nprofiles * num_counters * sizeof(long long));
      long long *sendbuf_excl = (long long*)malloc(nprofiles * num_counters * sizeof(long long));
      double *sendbuf_obs = (double*)malloc(nprofiles * n_observables * sizeof(double));

      for (int istack = 0; istack < nprofiles; istack++) {
         vftr_stack_t *mystack = stacktree_ptr->stacks + istack;
         gids[istack] = mystack->gid;
         profile_t *myprof = mystack->profiling.profiles;
         for (int e = 0; e < num_counters; e++) { 
            sendbuf_incl[istack * num_counters + e] = myprof->papiprof.counters_incl[e];
            sendbuf_excl[istack * num_counters + e] = myprof->papiprof.counters_excl[e];
         }
         for (int e = 0; e < n_observables; e++) {
            sendbuf_obs[istack * n_observables + e] = myprof->papiprof.observables[e];
         }
      }
      PMPI_Send (gids, nprofiles, MPI_INT, 0, myrank, MPI_COMM_WORLD);
      PMPI_Send (sendbuf_incl, nprofiles * num_counters, MPI_LONG_LONG, 0, myrank, MPI_COMM_WORLD);
      PMPI_Send (sendbuf_excl, nprofiles * num_counters, MPI_LONG_LONG, 0, myrank, MPI_COMM_WORLD);
      PMPI_Send (sendbuf_obs, nprofiles * n_observables, MPI_DOUBLE, 0, myrank, MPI_COMM_WORLD);
      free(gids);
      free(sendbuf_incl);
      free(sendbuf_excl);
      free(sendbuf_obs);
   } else {
      int maxprofiles = 0;
      for (int irank = 1; irank < nranks; irank++) {
         maxprofiles = nremote_profiles[irank] > maxprofiles ? nremote_profiles[irank] : maxprofiles;
      }

      for (int irank = 1; irank < nranks; irank++) {
         int nprofiles = nremote_profiles[irank];
         int *gids = (int*)malloc(nprofiles * sizeof(int));
         long long *recvbuf_incl = (long long*)malloc(nprofiles * num_counters * sizeof(long long));
         long long *recvbuf_excl = (long long*)malloc(nprofiles * num_counters * sizeof(long long));
         double *recvbuf_obs = (double*)malloc(nprofiles * n_observables * sizeof(double));
         MPI_Status status;
         PMPI_Recv (gids, nprofiles, MPI_INT, irank, irank, MPI_COMM_WORLD, &status);
         PMPI_Recv (recvbuf_incl, nprofiles * num_counters, MPI_LONG_LONG, irank, irank, MPI_COMM_WORLD, &status);
         PMPI_Recv (recvbuf_excl, nprofiles * num_counters, MPI_LONG_LONG, irank, irank, MPI_COMM_WORLD, &status);
         PMPI_Recv (recvbuf_obs, nprofiles * n_observables, MPI_DOUBLE, irank, irank, MPI_COMM_WORLD, &status);
         for (int iprof = 0; iprof < nprofiles; iprof++) {
            int gid = gids[iprof]; 
            collated_stack_t *collstack = collstacktree_ptr->stacks + gid;
            papiprofile_t *collpapiprof = &(collstack->profile.papiprof);

            for (int e = 0; e < num_counters; e++) {
               collpapiprof->counters_incl[e] += recvbuf_incl[iprof * num_counters + e];
               collpapiprof->counters_excl[e] += recvbuf_excl[iprof * num_counters + e];
            }
            for (int e = 0; e < n_observables; e++) {
               collpapiprof->observables[e] += recvbuf_obs[iprof * n_observables + e];
            }
         }
         free(gids);
         free(recvbuf_incl); 
         free(recvbuf_excl); 
         free(recvbuf_obs);
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
