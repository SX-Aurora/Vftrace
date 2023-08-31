#include <string.h>

#include "vftrace_state.h"

#include "collated_stack_types.h"
#include "cudaprofiling_types.h"
#include "collated_cudaprofiling_types.h"

void vftr_collate_cudaprofiles_root_self (collated_stacktree_t *collstacktree_ptr,
                                          stacktree_t *stacktree_ptr) {
   for (int istack = 0; istack < stacktree_ptr->nstacks; istack++) {
      vftr_stack_t *stack = stacktree_ptr->stacks + istack;
      int i_collstack = stack->gid;
      collated_stack_t *collstack = collstacktree_ptr->stacks + i_collstack;

      cudaprofile_t copy_cudaprof = stack->profiling.profiles[0].cudaprof;
      collated_cudaprofile_t *collcudaprof = &(collstack->profile.cudaprof);

      collcudaprof->cbid = copy_cudaprof.cbid;
      collcudaprof->n_calls[0] = copy_cudaprof.n_calls[0];
      collcudaprof->n_calls[1] = copy_cudaprof.n_calls[1];
      collcudaprof->t_ms = copy_cudaprof.t_ms;
      collcudaprof->memcpy_bytes[0] = copy_cudaprof.memcpy_bytes[0];
      collcudaprof->memcpy_bytes[1] = copy_cudaprof.memcpy_bytes[1];
      collcudaprof->overhead_nsec = copy_cudaprof.overhead_nsec;
   }
}

#ifdef _MPI
static void vftr_collate_cudaprofiles_on_root (collated_stacktree_t *collstacktree_ptr,
                                               stacktree_t *stacktree_ptr,
					       int myrank, int nranks, int *nremote_profiles) {

   typedef struct {
     int gid;
     int cbid;
     int n_calls;
     float t_ms;
     long long memcpy_bytes_in;
     long long memcpy_bytes_out;
     long long overhead_nsec;
   } cudaprofile_transfer_t;
   
   //3 blocks: 3 x int, 1 x float, 3 x long long
   int nblocks = 3;
   const int blocklengths[] = {3, 1, 3};
   const MPI_Aint displacements[] = {0, 3 * sizeof(int), 3 * sizeof(int) + sizeof(float)};
   const MPI_Datatype types[] = {MPI_INT, MPI_FLOAT, MPI_LONG_LONG_INT};
   MPI_Datatype cudaprofile_transfer_mpi_t;
   PMPI_Type_create_struct (nblocks, blocklengths, displacements, types,
                            &cudaprofile_transfer_mpi_t);
   PMPI_Type_commit (&cudaprofile_transfer_mpi_t);

   if (myrank > 0) {
      int nprofiles = stacktree_ptr->nstacks;
      cudaprofile_transfer_t *sendbuf = (cudaprofile_transfer_t*) malloc (nprofiles * sizeof(cudaprofile_transfer_t));
      memset (sendbuf, 0, nprofiles * sizeof(cudaprofile_transfer_t));
      for (int istack = 0; istack < nprofiles; istack++) {
         vftr_stack_t *mystack = stacktree_ptr->stacks + istack;
         cudaprofile_t cudaprof = mystack->profiling.profiles->cudaprof;
         sendbuf[istack].gid = mystack->gid;
         sendbuf[istack].cbid = cudaprof.cbid;
         sendbuf[istack].n_calls = cudaprof.n_calls;
         sendbuf[istack].t_ms = cudaprof.t_ms;
         sendbuf[istack].memcpy_bytes_in = cudaprof.memcpy_bytes[0];
         sendbuf[istack].memcpy_bytes_out = cudaprof.memcpy_bytes[1];
         sendbuf[istack].overhead_nsec = cudaprof.overhead_nsec;
      }
      PMPI_Send (sendbuf, nprofiles, cudaprofile_transfer_mpi_t, 0, myrank, MPI_COMM_WORLD);
      free(sendbuf);
   } else {
      int maxprofiles = 0;
      for (int irank = 1; irank < nranks; irank++) {
         maxprofiles = nremote_profiles[irank] > maxprofiles ? nremote_profiles[irank] : maxprofiles;
      }

      cudaprofile_transfer_t *recvbuf = (cudaprofile_transfer_t*)malloc(maxprofiles * sizeof(cudaprofile_transfer_t));
      memset (recvbuf, 0, maxprofiles * sizeof(cudaprofile_transfer_t));

      for (int irank = 1; irank < nranks; irank++) {
         int nprofiles = nremote_profiles[irank];
         MPI_Status status;
         PMPI_Recv (recvbuf, nprofiles, cudaprofile_transfer_mpi_t, irank, irank, MPI_COMM_WORLD, &status);
         for (int iprof = 0; iprof < nprofiles; iprof++) {
            int gid = recvbuf[iprof].gid;
            collated_stack_t *collstack = collstacktree_ptr->stacks + gid;
            collated_cudaprofile_t *collcudaprof = &(collstack->profile.cudaprof);

            collcudaprof->cbid = recvbuf[iprof].cbid;
            collcudaprof->n_calls += recvbuf[iprof].n_calls;
            collcudaprof->t_ms += recvbuf[iprof].t_ms;
            collcudaprof->memcpy_bytes[0] += recvbuf[iprof].memcpy_bytes_in;
            collcudaprof->memcpy_bytes[1] += recvbuf[iprof].memcpy_bytes_out;
            collcudaprof->overhead_nsec += recvbuf[iprof].overhead_nsec; 
         }
      }
      free(recvbuf);
   }
   PMPI_Type_free(&cudaprofile_transfer_mpi_t);
}
#endif

void vftr_collate_cudaprofiles (collated_stacktree_t *collstacktree_ptr,
			        stacktree_t *stacktree_ptr,
				int myrank, int nranks, int *nremote_profiles) {
   vftr_collate_cudaprofiles_root_self(collstacktree_ptr, stacktree_ptr);
#ifdef _MPI
   int mpi_initialized;
   PMPI_Initialized(&mpi_initialized);
   if (mpi_initialized) {
      vftr_collate_cudaprofiles_on_root (collstacktree_ptr, stacktree_ptr, myrank, nranks, nremote_profiles);
   }
#endif
}
