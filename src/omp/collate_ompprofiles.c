#include <stdlib.h>

#include <string.h>

#ifdef _MPI
#include <mpi.h>
#endif

#include "self_profile.h"
#include "ompprofiling_types.h"
#include "collated_stack_types.h"
#include "stack_types.h"

static void vftr_collate_ompprofiles_root_self(collated_stacktree_t *collstacktree_ptr,
                                               stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   for (int istack=0; istack<stacktree_ptr->nstacks; istack++) {
      stack_t *stack = stacktree_ptr->stacks+istack;
      int icollstack = stack->gid;

      collated_stack_t *collstack = collstacktree_ptr->stacks+icollstack;
      ompprofile_t *collompprof = &(collstack->profile.ompprof);

      collompprof->overhead_nsec = 0ll;

      for (int iprof=0; iprof<stack->profiling.nprofiles; iprof++) {
         ompprofile_t *ompprof = &(stack->profiling.profiles[iprof].ompprof);
   
         collompprof->overhead_nsec += ompprof->overhead_nsec;
      }
   }
   SELF_PROFILE_END_FUNCTION;
}

#ifdef _MPI
static void vftr_collate_ompprofiles_on_root(collated_stacktree_t *collstacktree_ptr,
                                             stacktree_t *stacktree_ptr,
                                             int myrank, int nranks,
                                             int *nremote_profiles) {
   SELF_PROFILE_START_FUNCTION;
   // define datatypes required for collating ompprofiles
   typedef struct {
      int gid;
      long long overhead_nsec;
   } ompprofile_transfer_t;

   int nblocks = 2;
   const int blocklengths[] = {1,1};
   const MPI_Aint displacements[] = {0, sizeof(int)};
   const MPI_Datatype types[] = {MPI_INT, MPI_LONG_LONG_INT};
   MPI_Datatype ompprofile_transfer_mpi_t;
   PMPI_Type_create_struct(nblocks, blocklengths,
                           displacements, types,
                           &ompprofile_transfer_mpi_t);
   PMPI_Type_commit(&ompprofile_transfer_mpi_t);

   if (myrank > 0) {
      // every rank fills their sendbuffer
      int nprofiles = stacktree_ptr->nstacks;
      ompprofile_transfer_t *sendbuf = (ompprofile_transfer_t*)
         malloc(nprofiles*sizeof(ompprofile_transfer_t));
      for (int istack=0; istack<nprofiles; istack++) {
         sendbuf[istack].gid = 0;
         sendbuf[istack].total_time_nsec = 0ll;
      }
      for (int istack=0; istack<nprofiles; istack++) {
         stack_t *mystack = stacktree_ptr->stacks+istack;
         sendbuf[istack].gid = mystack->gid;
         // need to go over the calling profiles threadwise
         for (int iprof=0; iprof<mystack->profiling.nprofiles; iprof++) {
            profile_t *myprof = mystack->profiling.profiles+iprof;
            ompprofile_t ompprof = myprof->ompprof;
            sendbuf[istack].overhead_nsec += ompprof.overhead_nsec;
         }
      }
      PMPI_Send(sendbuf, nprofiles,
                ompprofile_transfer_mpi_t,
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
      ompprofile_transfer_t *recvbuf = (ompprofile_transfer_t*)
         malloc(maxprofiles*sizeof(ompprofile_transfer_t));
      memset(recvbuf, 0, maxprofiles*sizeof(ompprofile_transfer_t));
      for (int irank=1; irank<nranks; irank++) {
         int nprofiles = nremote_profiles[irank];
         MPI_Status status;
         PMPI_Recv(recvbuf, nprofiles,
                   ompprofile_transfer_mpi_t,
                   irank, irank,
                   MPI_COMM_WORLD,
                   &status);
         for (int iprof=0; iprof<nprofiles; iprof++) {
            int gid = recvbuf[iprof].gid;
            collated_stack_t *collstack = collstacktree_ptr->stacks+gid;
            ompprofile_t *collompprof = &(collstack->profile.ompprof);
     
            collompprof->overhead_nsec += recvbuf[iprof].overhead_nsec;
         }
      }
      free(recvbuf);
   }

   PMPI_Type_free(&ompprofile_transfer_mpi_t);
   SELF_PROFILE_END_FUNCTION;
}
#endif

void vftr_collate_ompprofiles(collated_stacktree_t *collstacktree_ptr,
                              stacktree_t *stacktree_ptr,
                              int myrank, int nranks,
                              int *nremote_profiles) {
   SELF_PROFILE_START_FUNCTION;
   vftr_collate_ompprofiles_root_self(collstacktree_ptr, stacktree_ptr);
#ifdef _MPI
   int mpi_initialized;
   PMPI_Initialized(&mpi_initialized);
   if (mpi_initialized) {
      vftr_collate_ompprofiles_on_root(collstacktree_ptr, stacktree_ptr,
                                       myrank, nranks, nremote_profiles);
   }
#else
   (void) myrank;
   (void) nranks;
   (void) nremote_profiles;
#endif
   SELF_PROFILE_END_FUNCTION;
}
