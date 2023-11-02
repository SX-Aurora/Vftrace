#include <stdlib.h>

#include <string.h>

#ifdef _MPI
#include <mpi.h>
#endif

#include "self_profile.h"
#include "collated_callprofiling_types.h"
#include "collated_stack_types.h"
#include "stack_types.h"

static void vftr_collate_callprofiles_root_self(collated_stacktree_t *collstacktree_ptr,
                                                stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   for (int istack = 0; istack < stacktree_ptr->nstacks; istack++) {
      vftr_stack_t *stack = stacktree_ptr->stacks + istack;
      int icollstack = stack->gid;

      collated_stack_t *collstack = collstacktree_ptr->stacks + icollstack;
      collated_callprofile_t *collcallprof = &(collstack->profile.callprof);

      collcallprof->calls = 0ll;
      collcallprof->time_nsec = 0ll;
      collcallprof->time_excl_nsec = 0ll;
      collcallprof->overhead_nsec = 0ll;

      for (int iprof = 0; iprof < stack->profiling.nprofiles; iprof++) {
         callprofile_t *callprof = &(stack->profiling.profiles[iprof].callprof);
   
         collcallprof->calls += callprof->calls;
         collcallprof->time_nsec += callprof->time_nsec;
         collcallprof->time_excl_nsec += callprof->time_excl_nsec;
         collcallprof->overhead_nsec += callprof->overhead_nsec;
      }

      // call time imbalances info
      collcallprof->on_nranks = 1;
      collcallprof->max_on_rank = 0;
      collcallprof->min_on_rank = 0;
      collcallprof->average_time_nsec = collcallprof->time_excl_nsec;
      collcallprof->max_time_nsec = collcallprof->time_excl_nsec;
      collcallprof->min_time_nsec = collcallprof->time_excl_nsec;

   }
   SELF_PROFILE_END_FUNCTION;
}

#ifdef _MPI
static void vftr_collate_callprofiles_on_root(collated_stacktree_t *collstacktree_ptr,
                                              stacktree_t *stacktree_ptr,
                                              int myrank, int nranks,
                                              int *nremote_stacks) {
   SELF_PROFILE_START_FUNCTION;
   // define datatypes required for collating callprofiles
   typedef struct {
      int gid;
      long long calls;
      long long time_nsec;
      long long time_excl_nsec;
      long long overhead_nsec;
   } callprofile_transfer_t;

   int nblocks = 2;
   const int blocklengths[] = {1,4};
   const MPI_Aint displacements[] = {0, sizeof(int)};
   const MPI_Datatype types[] = {MPI_INT, MPI_LONG_LONG_INT};
   MPI_Datatype callprofile_transfer_mpi_t;
   PMPI_Type_create_struct(nblocks, blocklengths,
                           displacements, types,
                           &callprofile_transfer_mpi_t);
   PMPI_Type_commit(&callprofile_transfer_mpi_t);

   if (myrank > 0) {
      // every rank fills their sendbuffer
      int nstacks = stacktree_ptr->nstacks;
      callprofile_transfer_t *sendbuf = (callprofile_transfer_t*)
         malloc(nstacks * sizeof(callprofile_transfer_t));
      for (int istack = 0; istack < nstacks; istack++) {
         sendbuf[istack].gid = 0;
         sendbuf[istack].calls = 0ll;
         sendbuf[istack].time_nsec = 0ll;
         sendbuf[istack].time_excl_nsec = 0ll;
         sendbuf[istack].overhead_nsec = 0ll;
      }
      for (int istack = 0; istack < nstacks; istack++) {
         vftr_stack_t *mystack = stacktree_ptr->stacks+istack;
         sendbuf[istack].gid = mystack->gid;
         // need to go over the calling profiles threadwise
         for (int iprof = 0; iprof < mystack->profiling.nprofiles; iprof++) {
            profile_t *myprof = mystack->profiling.profiles+iprof;
            callprofile_t callprof = myprof->callprof;
            sendbuf[istack].calls += callprof.calls;
            sendbuf[istack].time_nsec += callprof.time_nsec;
            sendbuf[istack].time_excl_nsec += callprof.time_excl_nsec;
            sendbuf[istack].overhead_nsec += callprof.overhead_nsec;
         }
      }
      PMPI_Send(sendbuf, nstacks,
                callprofile_transfer_mpi_t,
                0, myrank,
                MPI_COMM_WORLD);
      free(sendbuf);
   } else {
      int maxstacks = 0;
      for (int irank = 1; irank < nranks; irank++) {
         maxstacks = nremote_stacks[irank] > maxstacks ? 
                       nremote_stacks[irank] :
                       maxstacks;
      }
      callprofile_transfer_t *recvbuf = (callprofile_transfer_t*)
         malloc(maxstacks * sizeof(callprofile_transfer_t));
      memset(recvbuf, 0, maxstacks * sizeof(callprofile_transfer_t));
      for (int irank = 1; irank < nranks; irank++) {
         int nstacks = nremote_stacks[irank];
         MPI_Status status;
         PMPI_Recv(recvbuf, nstacks,
                   callprofile_transfer_mpi_t,
                   irank, irank,
                   MPI_COMM_WORLD,
                   &status);
         for (int istack = 0; istack < nstacks; istack++) {
            int gid = recvbuf[istack].gid;
            collated_stack_t *collstack = collstacktree_ptr->stacks+gid;
            collated_callprofile_t *collcallprof = &(collstack->profile.callprof);
     
            collcallprof->calls += recvbuf[istack].calls;
            collcallprof->time_nsec += recvbuf[istack].time_nsec;
            collcallprof->time_excl_nsec += recvbuf[istack].time_excl_nsec;
            collcallprof->overhead_nsec += recvbuf[istack].overhead_nsec;

            // collect call time imbalances info
            if (recvbuf[istack].time_excl_nsec > 0) {
               collcallprof->on_nranks++;
               collcallprof->average_time_nsec += recvbuf[istack].time_excl_nsec;
               // search for min and max values
               if (recvbuf[istack].time_excl_nsec > collcallprof->max_time_nsec) {
                  collcallprof->max_on_rank = irank;
                  collcallprof->max_time_nsec = recvbuf[istack].time_excl_nsec;
               } else if (recvbuf[istack].time_excl_nsec < collcallprof->min_time_nsec) {
                  collcallprof->min_on_rank = irank;
                  collcallprof->min_time_nsec = recvbuf[istack].time_excl_nsec;
               }
            }
         }
      }
      free(recvbuf);
   }

   PMPI_Type_free(&callprofile_transfer_mpi_t);
   SELF_PROFILE_END_FUNCTION;
}
#endif

void vftr_compute_callprofile_imbalances(collated_stacktree_t *collstacktree_ptr) {
   for (int istack=0; istack<collstacktree_ptr->nstacks; istack++) {
      collated_stack_t *stack_ptr = collstacktree_ptr->stacks+istack;
      collated_callprofile_t *collcallprof_ptr = &(stack_ptr->profile.callprof);
      if (collcallprof_ptr->average_time_nsec > 0) {
         collcallprof_ptr->average_time_nsec /= collcallprof_ptr->on_nranks;
         double diff_from_max = collcallprof_ptr->max_time_nsec
                                - collcallprof_ptr->average_time_nsec;
         diff_from_max = diff_from_max < 0 ? -diff_from_max : diff_from_max;
         double diff_from_min = collcallprof_ptr->min_time_nsec
                                - collcallprof_ptr->average_time_nsec;
         diff_from_min = diff_from_min < 0 ? -diff_from_min : diff_from_min;
         if (diff_from_max > diff_from_min) {
            collcallprof_ptr->max_imbalance = 100.0*diff_from_max;
            collcallprof_ptr->max_imbalance /= collcallprof_ptr->average_time_nsec;
            collcallprof_ptr->max_imbalance_on_rank = collcallprof_ptr->max_on_rank;
         } else {
            collcallprof_ptr->max_imbalance = 100.0*diff_from_min;
            collcallprof_ptr->max_imbalance /= collcallprof_ptr->average_time_nsec;
            // if the imbalance will be different from 0% in the table (>0.005%)
            // indicate the direction of imbalance with a sign
            if (collcallprof_ptr->max_imbalance >= 0.005) {
               collcallprof_ptr->max_imbalance *= -1.0;
            }
            collcallprof_ptr->max_imbalance_on_rank = collcallprof_ptr->min_on_rank;
         }
      } else {
         collcallprof_ptr->max_imbalance = 0.0;
         collcallprof_ptr->max_imbalance_on_rank = 0;
      }
   }
}
void vftr_collate_callprofiles(collated_stacktree_t *collstacktree_ptr,
                               stacktree_t *stacktree_ptr,
                               int myrank, int nranks,
                               int *nremote_stacks) {
   SELF_PROFILE_START_FUNCTION;
   vftr_collate_callprofiles_root_self(collstacktree_ptr, stacktree_ptr);
#ifdef _MPI
   int mpi_initialized;
   PMPI_Initialized(&mpi_initialized);
   if (mpi_initialized) {
      vftr_collate_callprofiles_on_root(collstacktree_ptr, stacktree_ptr,
                                        myrank, nranks, nremote_stacks);
   }
#else
   (void) myrank;
   (void) nranks;
   (void) nremote_stacks;
#endif
   vftr_compute_callprofile_imbalances(collstacktree_ptr);
   SELF_PROFILE_END_FUNCTION;
}
