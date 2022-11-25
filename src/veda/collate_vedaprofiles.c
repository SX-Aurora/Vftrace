#include <stdlib.h>

#ifdef _MPI
#include <mpi.h>
#endif

#include "vedaprofiling_types.h"
#include "collated_stack_types.h"
#include "stack_types.h"

static void vftr_collate_vedaprofiles_root_self(collated_stacktree_t *collstacktree_ptr,
                                                stacktree_t *stacktree_ptr) {
   for (int istack=0; istack<stacktree_ptr->nstacks; istack++) {
      stack_t *stack = stacktree_ptr->stacks+istack;
      int icollstack = stack->gid;

      collated_stack_t *collstack = collstacktree_ptr->stacks+icollstack;
      vedaprofile_t *collvedaprof = &(collstack->profile.vedaprof);

      collvedaprof->ncalls = 0;
      collvedaprof->HtoD_bytes = 0ll;
      collvedaprof->DtoH_bytes = 0ll;
      collvedaprof->H_bytes = 0ll;
      collvedaprof->acc_HtoD_bw = 0.0;
      collvedaprof->acc_DtoH_bw = 0.0;
      collvedaprof->acc_H_bw = 0.0;
      collvedaprof->total_time_nsec = 0ll;
      collvedaprof->overhead_nsec = 0ll;

      for (int iprof=0; iprof<stack->profiling.nprofiles; iprof++) {
         vedaprofile_t *vedaprof = &(stack->profiling.profiles[iprof].vedaprof);

         collvedaprof->ncalls = vedaprof->ncalls;
         collvedaprof->HtoD_bytes = vedaprof->HtoD_bytes;
         collvedaprof->DtoH_bytes = vedaprof->DtoH_bytes;
         collvedaprof->H_bytes = vedaprof->H_bytes;
         collvedaprof->acc_HtoD_bw = vedaprof->acc_HtoD_bw;
         collvedaprof->acc_DtoH_bw = vedaprof->acc_DtoH_bw;
         collvedaprof->acc_H_bw = vedaprof->acc_H_bw;
         collvedaprof->total_time_nsec = vedaprof->total_time_nsec;
         collvedaprof->overhead_nsec = vedaprof->overhead_nsec;
      }
   }
}

#ifdef _MPI

static void vftr_collate_vedaprofiles_on_root(collated_stacktree_t *collstacktree_ptr,
                                              stacktree_t *stacktree_ptr,
                                              int myrank, int nranks,
                                              int *nremote_profiles) {
   // define datatypes required for collating mpiprofiles
   typedef struct {
      int gid;
      int ncalls;
      long long HtoD_bytes;
      long long DtoH_bytes;
      long long H_bytes;
      long long total_time_nsec;
      long long overhead_nsec;
      double acc_HtoD_bw;
      double acc_DtoH_bw;
      double acc_H_bw;
   } vedaprofile_transfer_t;

   int nblocks = 3;
   const int blocklengths[] = {2,5,3};
   const MPI_Aint displacements[] = {0,
                                     2*sizeof(int),
                                     2*sizeof(int)+5*sizeof(long long)};
   const MPI_Datatype types[] = {MPI_INT, MPI_LONG_LONG_INT, MPI_DOUBLE};
   MPI_Datatype vedaprofile_transfer_mpi_t;
   PMPI_Type_create_struct(nblocks, blocklengths,
                           displacements, types,
                           &vedaprofile_transfer_mpi_t);
   PMPI_Type_commit(&vedaprofile_transfer_mpi_t);

   if (myrank > 0) {
      // every rank fills their sendbuffer
      int nprofiles = stacktree_ptr->nstacks;
      vedaprofile_transfer_t *sendbuf = (vedaprofile_transfer_t*)
         malloc(nprofiles*sizeof(vedaprofile_transfer_t));
      for (int istack=0; istack<nprofiles; istack++) {
         sendbuf[istack].gid = 0;
         sendbuf[istack].ncalls = 0;
         sendbuf[istack].HtoD_bytes = 0ll;
         sendbuf[istack].DtoH_bytes = 0ll;
         sendbuf[istack].H_bytes = 0ll;
         sendbuf[istack].total_time_nsec = 0ll;
         sendbuf[istack].overhead_nsec = 0ll;
         sendbuf[istack].acc_HtoD_bw = 0.0;
         sendbuf[istack].acc_DtoH_bw = 0.0;
         sendbuf[istack].acc_H_bw = 0.0;
      }
      for (int istack=0; istack<nprofiles; istack++) {
         stack_t *mystack = stacktree_ptr->stacks+istack;
         sendbuf[istack].gid = mystack->gid;
         // need to go over the calling profiles threadwise
         for (int iprof=0; iprof<mystack->profiling.nprofiles; iprof++) {
            profile_t *myprof = mystack->profiling.profiles+iprof;
            vedaprofile_t vedaprof = myprof->vedaprof;

            sendbuf[istack].ncalls += vedaprof.ncalls;
            sendbuf[istack].HtoD_bytes += vedaprof.HtoD_bytes;
            sendbuf[istack].DtoH_bytes += vedaprof.DtoH_bytes;
            sendbuf[istack].H_bytes += vedaprof.H_bytes;
            sendbuf[istack].total_time_nsec += vedaprof.total_time_nsec;
            sendbuf[istack].overhead_nsec += vedaprof.overhead_nsec;
            sendbuf[istack].acc_HtoD_bw += vedaprof.acc_HtoD_bw;
            sendbuf[istack].acc_DtoH_bw += vedaprof.acc_DtoH_bw;
            sendbuf[istack].acc_H_bw += vedaprof.acc_H_bw;
         }
      }
      PMPI_Send(sendbuf, nprofiles,
                vedaprofile_transfer_mpi_t,
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
      vedaprofile_transfer_t *recvbuf = (vedaprofile_transfer_t*)
         malloc(maxprofiles*sizeof(vedaprofile_transfer_t));
      memset(recvbuf, 0, maxprofiles*sizeof(vedaprofile_transfer_t));
      for (int irank=1; irank<nranks; irank++) {
         int nprofiles = nremote_profiles[irank];
         MPI_Status status;
         PMPI_Recv(recvbuf, nprofiles,
                   vedaprofile_transfer_mpi_t,
                   irank, irank,
                   MPI_COMM_WORLD,
                   &status);
         for (int iprof=0; iprof<nprofiles; iprof++) {
            int gid = recvbuf[iprof].gid;
            collated_stack_t *collstack = collstacktree_ptr->stacks+gid;
            vedaprofile_t *collvedaprof = &(collstack->profile.vedaprof);


            collvedaprof->ncalls += recvbuf[iprof].ncalls;
            collvedaprof->HtoD_bytes += recvbuf[iprof].HtoD_bytes;
            collvedaprof->DtoH_bytes += recvbuf[iprof].DtoH_bytes;
            collvedaprof->H_bytes += recvbuf[iprof].H_bytes;
            collvedaprof->total_time_nsec += recvbuf[iprof].total_time_nsec;
            collvedaprof->overhead_nsec += recvbuf[iprof].overhead_nsec;
            collvedaprof->acc_HtoD_bw += recvbuf[iprof].acc_HtoD_bw;
            collvedaprof->acc_DtoH_bw += recvbuf[iprof].acc_DtoH_bw;
            collvedaprof->acc_H_bw += recvbuf[iprof].acc_H_bw;
         }
      }
      free(recvbuf);
   }

   PMPI_Type_free(&vedaprofile_transfer_mpi_t);
}
#endif

void vftr_collate_vedaprofiles(collated_stacktree_t *collstacktree_ptr,
                               stacktree_t *stacktree_ptr,
                               int myrank, int nranks,
                               int *nremote_profiles) {
   vftr_collate_vedaprofiles_root_self(collstacktree_ptr, stacktree_ptr);
#ifdef _MPI
   int mpi_initialized;
   PMPI_Initialized(&mpi_initialized);
   if (mpi_initialized) {
      vftr_collate_vedaprofiles_on_root(collstacktree_ptr, stacktree_ptr,
                                        myrank, nranks, nremote_profiles);
   }
#else
   (void) myrank;
   (void) nranks;
   (void) nremote_profiles;
#endif
}


