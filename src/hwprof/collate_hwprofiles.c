#include <string.h>

#ifdef _MPI
#include "status_utils.h"
#endif

#include "vftrace_state.h"
#include "collated_stack_types.h"
#include "hwprofiling_types.h"

void vftr_collate_hwprofiles_root_self (collated_stacktree_t *collstacktree_ptr,
			                  stacktree_t *stacktree_ptr) {
   for (int istack = 0; istack < stacktree_ptr->nstacks; istack++) {
      vftr_stack_t *stack = stacktree_ptr->stacks + istack;
      int i_collstack = stack->gid;
      collated_stack_t *collstack = collstacktree_ptr->stacks + i_collstack;

      hwprofile_t copy_hwprof = stack->profiling.profiles[0].hwprof;
      hwprofile_t *collhwprof = &(collstack->profile.hwprof);
     
      int n_counters = vftrace.hwprof_state.n_counters;
      int n_observables = vftrace.hwprof_state.n_observables;

      if (n_counters > 0) {
         collhwprof->counters_incl = (long long*)malloc (n_counters * sizeof(long long));
         collhwprof->counters_excl = (long long*)malloc (n_counters * sizeof(long long));
         memcpy (collhwprof->counters_incl, copy_hwprof.counters_incl, n_counters * sizeof(long long));
         memcpy (collhwprof->counters_excl, copy_hwprof.counters_excl, n_counters * sizeof(long long));
      }
      if (n_observables > 0) {
         collhwprof->observables = (double*)malloc (n_observables * sizeof(double));
         memcpy (collhwprof->observables, copy_hwprof.observables, n_observables * sizeof(double));
      }
   }
}

#ifdef _MPI
static void vftr_collate_hwprofiles_on_root (collated_stacktree_t *collstacktree_ptr,
                                               stacktree_t *stacktree_ptr,
					       int myrank, int nranks, int *nremote_profiles) {

   typedef struct {
      long long counter_incl;
      long long counter_excl;
      double observable;
      int gid;
   } hwprofile_transfer_t;

   int num_counters = vftrace.hwprof_state.n_counters;
   int n_observables = vftrace.hwprof_state.n_observables;

   int nblocks = 3;
   const int blocklengths[] = {2, 1, 1};
   const MPI_Aint displacements[] = {0, 2 * sizeof(long long), sizeof(double) + 2 * sizeof(long long)};
   const MPI_Datatype types[] = {MPI_LONG_LONG_INT, MPI_DOUBLE, MPI_INT};
   MPI_Datatype hwprofile_transfer_mpi_t;
   PMPI_Type_create_struct (nblocks, blocklengths, displacements, types,
                            &hwprofile_transfer_mpi_t);
   PMPI_Type_commit (&hwprofile_transfer_mpi_t);

   int n_send = num_counters > n_observables ? num_counters : n_observables;
   if (n_send == 0) return;

   if (myrank > 0) {
      int nprofiles = stacktree_ptr->nstacks;
      hwprofile_transfer_t *sendbuf = (hwprofile_transfer_t*)malloc(nprofiles * sizeof(hwprofile_transfer_t));
      for (int isend = 0; isend < n_send; isend++) {
         for (int istack = 0; istack < nprofiles; istack++) {
            vftr_stack_t *mystack = stacktree_ptr->stacks + istack;
            hwprofile_t hwprof = mystack->profiling.profiles[0].hwprof;
            sendbuf[istack].gid = mystack->gid;
            sendbuf[istack].counter_incl = isend < num_counters ? hwprof.counters_incl[isend] : 0;
            sendbuf[istack].counter_excl = isend < num_counters ? hwprof.counters_excl[isend] : 0;
            sendbuf[istack].observable = isend < n_observables ? hwprof.observables[isend] : 0;
         }

         PMPI_Send (sendbuf, nprofiles, hwprofile_transfer_mpi_t, 0, myrank, MPI_COMM_WORLD);
      }
      free(sendbuf);
   } else {
      int maxprofiles = 0;
      for (int irank = 1; irank < nranks; irank++) {
         maxprofiles = nremote_profiles[irank] > maxprofiles ? nremote_profiles[irank] : maxprofiles;
      }

      hwprofile_transfer_t *recvbuf = (hwprofile_transfer_t*)malloc(maxprofiles * sizeof(hwprofile_transfer_t));

      for (int isend = 0; isend < n_send; isend++) {
         for (int irank = 1; irank < nranks; irank++) {
            int nprofiles = nremote_profiles[irank];
            MPI_Status status;
            PMPI_Recv (recvbuf, nprofiles, hwprofile_transfer_mpi_t, irank, irank, MPI_COMM_WORLD, &status);
            for (int iprof = 0; iprof < nprofiles; iprof++) {
               int gid = recvbuf[iprof].gid;
               collated_stack_t *collstack = collstacktree_ptr->stacks + gid;
               hwprofile_t *collhwprof = &(collstack->profile.hwprof);

               if (isend < num_counters) collhwprof->counters_incl[isend] += recvbuf[iprof].counter_incl;
               if (isend < num_counters) collhwprof->counters_excl[isend] += recvbuf[iprof].counter_excl;
               if (isend < n_observables) collhwprof->observables[isend] += recvbuf[iprof].observable;
            }
         }
      }
      free(recvbuf);
   }
   PMPI_Type_free(&hwprofile_transfer_mpi_t);
}
#endif

void vftr_collate_hwprofiles (collated_stacktree_t *collstacktree_ptr,
			        stacktree_t *stacktree_ptr,
				int myrank, int nranks, int *nremote_profiles) {
   vftr_collate_hwprofiles_root_self(collstacktree_ptr, stacktree_ptr);
#ifdef _MPI
   int mpi_initialized;
   PMPI_Initialized(&mpi_initialized);
   if (mpi_initialized) {
      vftr_collate_hwprofiles_on_root (collstacktree_ptr, stacktree_ptr, myrank, nranks, nremote_profiles);
   }
#endif
}
