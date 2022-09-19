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
   for (int istack=0; istack<stacktree_ptr->nstacks; istack++) {
      stack_t *stack = stacktree_ptr->stacks+istack;
      int icollstack = stack->gid;

      collated_stack_t *collstack = collstacktree_ptr->stacks+icollstack;
      collated_callProfile_t *collcallprof = &(collstack->profile.callProf);

      collcallprof->calls = 0ll;
      collcallprof->time_nsec = 0ll;
      collcallprof->time_excl_nsec = 0ll;
      collcallprof->overhead_nsec = 0ll;

      for (int iprof=0; iprof<stack->profiling.nprofiles; iprof++) {
         callProfile_t *callprof = &(stack->profiling.profiles[iprof].callProf);
   
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
                                              int *nremote_profiles) {
   SELF_PROFILE_START_FUNCTION;
   // define datatypes required for collating callprofiles
   typedef struct {
      int gid;
      long long calls;
      long long time_nsec;
      long long time_excl_nsec;
      long long overhead_nsec;
   } callProfile_transfer_t;

   int nblocks = 2;
   const int blocklengths[] = {1,4};
   const MPI_Aint displacements[] = {0, sizeof(int)};
   const MPI_Datatype types[] = {MPI_INT, MPI_LONG_LONG_INT};
   MPI_Datatype callProfile_transfer_mpi_t;
   PMPI_Type_create_struct(nblocks, blocklengths,
                           displacements, types,
                           &callProfile_transfer_mpi_t);
   PMPI_Type_commit(&callProfile_transfer_mpi_t);

   if (myrank > 0) {
      // every rank fills their sendbuffer
      int nprofiles = stacktree_ptr->nstacks;
      callProfile_transfer_t *sendbuf = (callProfile_transfer_t*)
         malloc(nprofiles*sizeof(callProfile_transfer_t));
      for (int istack=0; istack<nprofiles; istack++) {
         sendbuf[istack].gid = 0;
         sendbuf[istack].calls = 0ll;
         sendbuf[istack].time_nsec = 0ll;
         sendbuf[istack].time_excl_nsec = 0ll;
         sendbuf[istack].overhead_nsec = 0ll;
      }
      for (int istack=0; istack<nprofiles; istack++) {
         stack_t *mystack = stacktree_ptr->stacks+istack;
         sendbuf[istack].gid = mystack->gid;
         // need to go over the calling profiles threadwise
         for (int iprof=0; iprof<mystack->profiling.nprofiles; iprof++) {
            profile_t *myprof = mystack->profiling.profiles+iprof;
            callProfile_t callprof = myprof->callProf;
            sendbuf[istack].calls += callprof.calls;
            sendbuf[istack].time_nsec += callprof.time_nsec;
            sendbuf[istack].time_excl_nsec += callprof.time_excl_nsec;
            sendbuf[istack].overhead_nsec += callprof.overhead_nsec;
         }
      }
      PMPI_Send(sendbuf, nprofiles,
                callProfile_transfer_mpi_t,
                0, myrank,
                MPI_COMM_WORLD);
      free(sendbuf);
   } else {
      int maxprofiles = 0;
      for (int irank=1; irank<nranks; irank++) {
         maxprofiles = nremote_profiles[irank] > maxprofiles ? 
                       nremote_profiles[irank] :
                       maxprofiles;
      }
      callProfile_transfer_t *recvbuf = (callProfile_transfer_t*)
         malloc(maxprofiles*sizeof(callProfile_transfer_t));
      memset(recvbuf, 0, maxprofiles*sizeof(callProfile_transfer_t));
      for (int irank=1; irank<nranks; irank++) {
         int nprofiles = nremote_profiles[irank];
         MPI_Status status;
         PMPI_Recv(recvbuf, nprofiles,
                   callProfile_transfer_mpi_t,
                   irank, irank,
                   MPI_COMM_WORLD,
                   &status);
         for (int iprof=0; iprof<nprofiles; iprof++) {
            int gid = recvbuf[iprof].gid;
            collated_stack_t *collstack = collstacktree_ptr->stacks+gid;
            collated_callProfile_t *collcallprof = &(collstack->profile.callProf);
     
            collcallprof->calls += recvbuf[iprof].calls;
            collcallprof->time_nsec += recvbuf[iprof].time_nsec;
            collcallprof->time_excl_nsec += recvbuf[iprof].time_excl_nsec;
            collcallprof->overhead_nsec += recvbuf[iprof].overhead_nsec;

            // collect call time imbalances info
            if (recvbuf[iprof].time_excl_nsec > 0) {
               collcallprof->on_nranks++;
               collcallprof->average_time_nsec += recvbuf[iprof].time_excl_nsec;
               // search for min and max values
               if (recvbuf[iprof].time_excl_nsec > collcallprof->max_time_nsec) {
                  collcallprof->max_on_rank = irank;
                  collcallprof->max_time_nsec = recvbuf[iprof].time_excl_nsec;
               } else if (recvbuf[iprof].time_excl_nsec < collcallprof->min_time_nsec) {
                  collcallprof->min_on_rank = irank;
                  collcallprof->min_time_nsec = recvbuf[iprof].time_excl_nsec;
               }
            }
         }
      }
      free(recvbuf);
   }

   PMPI_Type_free(&callProfile_transfer_mpi_t);
   SELF_PROFILE_END_FUNCTION;
}
#endif

void vftr_compute_callprofile_imbalances(collated_stacktree_t *collstacktree_ptr) {
   for (int istack=0; istack<collstacktree_ptr->nstacks; istack++) {
      collated_stack_t *stack_ptr = collstacktree_ptr->stacks+istack;
      collated_callProfile_t *collcallProf_ptr = &(stack_ptr->profile.callProf);
      if (collcallProf_ptr->average_time_nsec > 0) {
         collcallProf_ptr->average_time_nsec /= collcallProf_ptr->on_nranks;
         double diff_from_max = collcallProf_ptr->max_time_nsec
                                - collcallProf_ptr->average_time_nsec;
         diff_from_max = diff_from_max < 0 ? -diff_from_max : diff_from_max;
         double diff_from_min = collcallProf_ptr->min_time_nsec
                                - collcallProf_ptr->average_time_nsec;
         diff_from_min = diff_from_min < 0 ? -diff_from_min : diff_from_min;
         if (diff_from_max > diff_from_min) {
            collcallProf_ptr->max_imbalance = 100.0*diff_from_max;
            collcallProf_ptr->max_imbalance /= collcallProf_ptr->average_time_nsec;
            collcallProf_ptr->max_imbalance_on_rank = collcallProf_ptr->max_on_rank;
         } else {
            collcallProf_ptr->max_imbalance = 100.0*diff_from_min;
            collcallProf_ptr->max_imbalance /= collcallProf_ptr->average_time_nsec;
            collcallProf_ptr->max_imbalance_on_rank = collcallProf_ptr->min_on_rank;
         }
      } else {
         collcallProf_ptr->max_imbalance = 0.0;
         collcallProf_ptr->max_imbalance_on_rank = 0;
      }
   }
}
void vftr_collate_callprofiles(collated_stacktree_t *collstacktree_ptr,
                               stacktree_t *stacktree_ptr,
                               int myrank, int nranks,
                               int *nremote_profiles) {
   SELF_PROFILE_START_FUNCTION;
   vftr_collate_callprofiles_root_self(collstacktree_ptr, stacktree_ptr);
#ifdef _MPI
   vftr_collate_callprofiles_on_root(collstacktree_ptr, stacktree_ptr,
                                     myrank, nranks, nremote_profiles);
#else
   (void) myrank;
   (void) nranks;
   (void) nremote_profiles;
#endif
   vftr_compute_callprofile_imbalances(collstacktree_ptr);
   SELF_PROFILE_END_FUNCTION;
}
