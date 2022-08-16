#include <stdlib.h>

#include <string.h>
#include <mpi.h>

#include "mpiprofiling_types.h"
#include "collated_stack_types.h"
#include "stack_types.h"

static void vftr_collate_mpiprofiles_root_self(collated_stacktree_t *collstacktree_ptr,
                                               stacktree_t *stacktree_ptr) {
   for (int istack=0; istack<stacktree_ptr->nstacks; istack++) {
      stack_t *stack = stacktree_ptr->stacks+istack;
      int icollstack = stack->gid;

      collated_stack_t *collstack = collstacktree_ptr->stacks+icollstack;
      mpiProfile_t *collmpiprof = &(collstack->profile.mpiProf);

      collmpiprof->nsendmessages = 0ll;
      collmpiprof->nrecvmessages = 0ll;
      collmpiprof->send_bytes = 0ll;
      collmpiprof->recv_bytes = 0ll;
      collmpiprof->acc_send_bw = 0.0;
      collmpiprof->acc_recv_bw = 0.0;
      collmpiprof->total_time_usec = 0ll;
      collmpiprof->overhead_usec = 0ll;

      for (int iprof=0; iprof<stack->profiling.nprofiles; iprof++) {
         mpiProfile_t *mpiprof = &(stack->profiling.profiles[iprof].mpiProf);
   
         collmpiprof->nsendmessages += mpiprof->nsendmessages;
         collmpiprof->nrecvmessages += mpiprof->nrecvmessages;
         collmpiprof->send_bytes += mpiprof->send_bytes;
         collmpiprof->recv_bytes += mpiprof->recv_bytes;
         collmpiprof->acc_send_bw += mpiprof->acc_send_bw;
         collmpiprof->acc_recv_bw += mpiprof->acc_recv_bw;
         collmpiprof->total_time_usec += mpiprof->total_time_usec;
         collmpiprof->overhead_usec += mpiprof->overhead_usec;
      }
   }
}

static void vftr_collate_mpiprofiles_on_root(collated_stacktree_t *collstacktree_ptr,
                                             stacktree_t *stacktree_ptr,
                                             int myrank, int nranks,
                                             int *nremote_profiles) {
   // define datatypes required for collating mpiprofiles
   typedef struct {
      int gid;
      long long nsendmessages;
      long long nrecvmessages;
      long long send_bytes;
      long long recv_bytes;
      long long total_time_usec;
      long long overhead_usec;
      double acc_send_bw;
      double acc_recv_bw;
   } mpiProfile_transfer_t;

   int nblocks = 3;
   const int blocklengths[] = {1,6,2};
   const MPI_Aint displacements[] = {0,
                                     sizeof(int),
                                     sizeof(int)+6*sizeof(long long)};
   const MPI_Datatype types[] = {MPI_INT, MPI_LONG_LONG_INT, MPI_DOUBLE};
   MPI_Datatype mpiProfile_transfer_mpi_t;
   PMPI_Type_create_struct(nblocks, blocklengths,
                           displacements, types,
                           &mpiProfile_transfer_mpi_t);
   PMPI_Type_commit(&mpiProfile_transfer_mpi_t);

   if (myrank > 0) {
      // every rank fills their sendbuffer
      int nprofiles = stacktree_ptr->nstacks;
      mpiProfile_transfer_t *sendbuf = (mpiProfile_transfer_t*)
         malloc(nprofiles*sizeof(mpiProfile_transfer_t));
      for (int istack=0; istack<nprofiles; istack++) {
         sendbuf[istack].gid = 0;
         sendbuf[istack].nsendmessages = 0ll;
         sendbuf[istack].nrecvmessages = 0ll;
         sendbuf[istack].send_bytes = 0ll;
         sendbuf[istack].recv_bytes = 0ll;
         sendbuf[istack].total_time_usec = 0ll;
         sendbuf[istack].overhead_usec = 0ll;
         sendbuf[istack].acc_send_bw = 0.0;
         sendbuf[istack].acc_recv_bw = 0.0;
      }
      for (int istack=0; istack<nprofiles; istack++) {
         stack_t *mystack = stacktree_ptr->stacks+istack;
         sendbuf[istack].gid = mystack->gid;
         // need to go over the calling profiles threadwise
         for (int iprof=0; iprof<mystack->profiling.nprofiles; iprof++) {
            profile_t *myprof = mystack->profiling.profiles+iprof;
            mpiProfile_t mpiprof = myprof->mpiProf;

            sendbuf[istack].nsendmessages += mpiprof.nsendmessages;
            sendbuf[istack].nrecvmessages += mpiprof.nrecvmessages;
            sendbuf[istack].send_bytes += mpiprof.send_bytes;
            sendbuf[istack].recv_bytes += mpiprof.recv_bytes;
            sendbuf[istack].acc_send_bw += mpiprof.acc_send_bw;
            sendbuf[istack].acc_recv_bw += mpiprof.acc_recv_bw;
            sendbuf[istack].total_time_usec += mpiprof.total_time_usec;
            sendbuf[istack].overhead_usec += mpiprof.overhead_usec;

         }
      }
      PMPI_Send(sendbuf, nprofiles,
                mpiProfile_transfer_mpi_t,
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
      mpiProfile_transfer_t *recvbuf = (mpiProfile_transfer_t*)
         malloc(maxprofiles*sizeof(mpiProfile_transfer_t));
      memset(recvbuf, 0, maxprofiles*sizeof(mpiProfile_transfer_t));
      for (int irank=1; irank<nranks; irank++) {
         int nprofiles = nremote_profiles[irank];
         MPI_Status status;
         PMPI_Recv(recvbuf, nprofiles,
                   mpiProfile_transfer_mpi_t,
                   irank, irank,
                   MPI_COMM_WORLD,
                   &status);
         for (int iprof=0; iprof<nprofiles; iprof++) {
            int gid = recvbuf[iprof].gid;
            collated_stack_t *collstack = collstacktree_ptr->stacks+gid;
            mpiProfile_t *collmpiprof = &(collstack->profile.mpiProf);
     
            collmpiprof->nsendmessages += recvbuf[iprof].nsendmessages;
            collmpiprof->nrecvmessages += recvbuf[iprof].nrecvmessages;
            collmpiprof->send_bytes += recvbuf[iprof].send_bytes;
            collmpiprof->recv_bytes += recvbuf[iprof].recv_bytes;
            collmpiprof->acc_send_bw += recvbuf[iprof].acc_send_bw;
            collmpiprof->acc_recv_bw += recvbuf[iprof].acc_recv_bw;
            collmpiprof->total_time_usec += recvbuf[iprof].total_time_usec;
            collmpiprof->overhead_usec += recvbuf[iprof].overhead_usec;
         }
      }
      free(recvbuf);
   }

   PMPI_Type_free(&mpiProfile_transfer_mpi_t);
}

void vftr_collate_mpiprofiles(collated_stacktree_t *collstacktree_ptr,
                              stacktree_t *stacktree_ptr,
                              int myrank, int nranks,
                              int *nremote_profiles) {
   vftr_collate_mpiprofiles_root_self(collstacktree_ptr, stacktree_ptr);
   vftr_collate_mpiprofiles_on_root(collstacktree_ptr, stacktree_ptr,
                                    myrank, nranks, nremote_profiles);
}
