#include <string.h>

#ifdef _MPI
#include "status_utils.h"
#endif

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
     
      int n_counters = vftrace.papi_state.n_counters;
      int n_observables = vftrace.config.papi.observables.obs_name.n_elements;
      collpapiprof->counters_incl = (long long*)malloc (n_counters * sizeof(long long));
      collpapiprof->counters_excl = (long long*)malloc (n_counters * sizeof(long long));
      collpapiprof->observables = (double*)malloc (n_observables * sizeof(double));
      memcpy (collpapiprof->counters_incl, copy_papiprof.counters_incl, n_counters * sizeof(long long));
      memcpy (collpapiprof->counters_excl, copy_papiprof.counters_excl, n_counters * sizeof(long long));
      memcpy (collpapiprof->observables, copy_papiprof.observables, n_observables * sizeof(double));
   }
}

#ifdef _MPI
static void vftr_collate_papiprofiles_on_root (collated_stacktree_t *collstacktree_ptr,
                                               stacktree_t *stacktree_ptr,
					       int myrank, int nranks, int *nremote_profiles) {

   typedef struct {
      int gid;
      long long counter_incl;
      long long counter_excl;
      double observable;
   } papiprofile_transfer_t;

   int num_counters = vftrace.papi_state.n_counters;
   int n_observables = vftrace.config.papi.observables.obs_name.n_elements;

   int nblocks = 3;
   const int blocklengths[] = {1, 2, 1};
   const MPI_Aint displacements[] = {0, sizeof(int), sizeof(int) + 2 * sizeof(long long)};
   const MPI_Datatype types[] = {MPI_INT, MPI_LONG_LONG_INT, MPI_DOUBLE};
   MPI_Datatype papiprofile_transfer_mpi_t;
   PMPI_Type_create_struct (nblocks, blocklengths, displacements, types,
                            &papiprofile_transfer_mpi_t);
   PMPI_Type_commit (&papiprofile_transfer_mpi_t);

   int n_send = num_counters > n_observables ? num_counters : n_observables;

   if (myrank > 0) {
      int nprofiles = stacktree_ptr->nstacks;
      papiprofile_transfer_t *sendbuf = (papiprofile_transfer_t*)malloc(nprofiles * sizeof(papiprofile_transfer_t));
      for (int isend = 0; isend < n_send; isend++) {
         for (int istack = 0; istack < nprofiles; istack++) {
            vftr_stack_t *mystack = stacktree_ptr->stacks + istack;
            papiprofile_t papiprof = mystack->profiling.profiles->papiprof;
            sendbuf[istack].gid = mystack->gid;
            sendbuf[istack].counter_incl = isend < num_counters ? papiprof.counters_incl[isend] : 0;
            sendbuf[istack].counter_excl = isend < num_counters ? papiprof.counters_excl[isend] : 0;
            sendbuf[istack].observable = isend < n_observables ? papiprof.observables[isend] : 0;
         }

         PMPI_Send (sendbuf, nprofiles, papiprofile_transfer_mpi_t, 0, myrank, MPI_COMM_WORLD);
      }
      free(sendbuf);
   } else {
      int maxprofiles = 0;
      for (int irank = 1; irank < nranks; irank++) {
         maxprofiles = nremote_profiles[irank] > maxprofiles ? nremote_profiles[irank] : maxprofiles;
      }

      papiprofile_transfer_t *recvbuf = (papiprofile_transfer_t*)malloc(maxprofiles * sizeof(papiprofile_transfer_t));

      for (int isend = 0; isend < n_send; isend++) {
         for (int irank = 1; irank < nranks; irank++) {
            int nprofiles = nremote_profiles[irank];
            MPI_Status status;
            PMPI_Recv (recvbuf, nprofiles, papiprofile_transfer_mpi_t, irank, irank, MPI_COMM_WORLD, &status);
            for (int iprof = 0; iprof < nprofiles; iprof++) {
               int gid = recvbuf[iprof].gid;
               collated_stack_t *collstack = collstacktree_ptr->stacks + gid;
               papiprofile_t *collpapiprof = &(collstack->profile.papiprof);

               if (isend < num_counters) collpapiprof->counters_incl[isend] += recvbuf[iprof].counter_incl;
               if (isend < num_counters) collpapiprof->counters_excl[isend] += recvbuf[iprof].counter_excl;
               if (isend < n_observables) collpapiprof->observables[isend] += recvbuf[iprof].observable;
            }
         }
      }
      free(recvbuf);
   }
   PMPI_Type_free(&papiprofile_transfer_mpi_t);
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
