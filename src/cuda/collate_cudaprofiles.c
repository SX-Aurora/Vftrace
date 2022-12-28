#include "collated_stack_types.h"
#include "cudaprofiling_types.h"
#include "collated_cudaprofiling_types.h"

// Currently, CUDA profiling is only supported for
// one MPI process and one OMP thread. Therefore, collating
// the profiles just comes down to copying the profile from
// the one stack which exists.
//
void vftr_collate_cudaprofiles_root_self (collated_stacktree_t *collstacktree_ptr,
                                          stacktree_t *stacktree_ptr) {
   for (int istack = 0; istack < stacktree_ptr->nstacks; istack++) {
      vftr_stack_t *stack = stacktree_ptr->stacks + istack;
      int i_collstack = stack->gid;
      collated_stack_t *collstack = collstacktree_ptr->stacks + i_collstack;

      cudaprofile_t copy_cudaprof = stack->profiling.profiles[0].cudaprof;
      collated_cudaprofile_t *collcudaprof = &(collstack->profile.cudaprof);

      collcudaprof->cbid = copy_cudaprof.cbid;
      collcudaprof->n_calls = copy_cudaprof.n_calls;
      collcudaprof->t_ms = copy_cudaprof.t_ms;
      //printf ("memcpy_bytes: %lld %lld\n", copy_cudaprof.memcpy_bytes[0], copy_cudaprof.memcpy_bytes[1]);
      //collcudaprof->memcpy_bytes[0] = copy_cudaprof.memcpy_bytes[0];
      //collcudaprof->memcpy_bytes[1] = copy_cudaprof.memcpy_bytes[1];
      collcudaprof->overhead_nsec = copy_cudaprof.overhead_nsec;
   }
}

#ifdef _MPI
static void vftr_collate_papiprofiles_on_root (collated_stacktree_t *collstacktree_ptr,
                                               stacktree_t *stacktree_ptr,
					       int myrank, int nranks, int *nremote_profiles) {

   if (myrank > 0) {
      int nprofiles = stacktree_ptr->nstacks;
      int *gids = (int*)malloc(nprofiles * sizeof(int));
      int *cbids = (int*) malloc(nprofiles * sizeof(int));
      int *n_calls = (int*)malloc(nprofiles * sizeof(int));
      float *t_ms = (float*)malloc(nprofiles * sizeof(float));
      size_t *memcpy_bytes_1 = (size_t*)malloc(nprofiles * sizeof(size_t));
      size_t *memcpy_bytes_2 = (size_t*)malloc(nprofiles * sizeof(size_t));
      long long *overhead_nsec = (long long*)malloc(nprofiles * sizeof(long long));

      for (int istack = 0; istack < nprofiles; istack++) {
         vftr_stack_t *mystack = stacktree_ptr->stacks + istack;
         gids[istack] = mystack->gid;
         profile_t *myprof = mystack->profiling.profiles;
         cbids[istack] = myprof->cudaprof.cbid;
         n_calls[istack] = myprof->cudaprof.n_calls;
         t_ms[istack] = myprof->cudaprof.t_ms;
         memcpy_bytes_1[istack] = myprof->cudaprof.memcpy_bytes[0];
         memcpy_bytes_2[istack] = myprof->cudaprof.memcpy_bytes[1];
         overhead_nsec[istack] = myprof->cudaprof.overhead_nsec;
      }
      PMPI_Send (gids, nprofiles, MPI_INT, 0, myrank, MPI_COMM_WORLD);
      PMPI_Send (cbids, nprofiles, MPI_INT, 0, myrank, MPI_COMM_WORLD);
      PMPI_Send (n_calls, nprofiles, MPI_INT, 0, myrank, MPI_COMM_WORLD);
      PMPI_Send (t_ms, nprofiles, MPI_FLOAT, 0, myrank, MPI_COMM_WORLD);
      PMPI_Send (memcpy_bytes_1, MPI_LONG, 0, myrank, MPI_COMM_WORLD);
      PMPI_Send (memcpy_bytes_2, MPI_LONG, 0, myrank, MPI_COMM_WORLD);
      PMPI_Send (overhead_nsec, MPI_LONG_LONG, 0, myrank, MPI_COMM_WORLD);
      free(gids);
      free(cbids);
      free(n_calls);
      free(t_ms);
      free(memcpy_bytes_1);
      free(memcpy_bytes_2);
      free(overhead_nsec);
   } else {
      int maxprofiles = 0;
      for (int irank = 1; irank < nranks; irank++) {
         maxprofiles = nremote_profiles[irank] > maxprofiles ? nremote_profiles[irank] : maxprofiles;
      }

      for (int irank = 1; irank < nranks; irank++) {
         int nprofiles = nremote_profiles[irank];
         int *gids = (int*)malloc(nprofiles * sizeof(int));
         int *cbids = (int*) malloc(nprofiles * sizeof(int));
         int *n_calls = (int*)malloc(nprofiles * sizeof(int));
         float *t_ms = (float*)malloc(nprofiles * sizeof(float));
         size_t *memcpy_bytes_1 = (size_t*)malloc(nprofiles * sizeof(size_t));
         size_t *memcpy_bytes_2 = (size_t*)malloc(nprofiles * sizeof(size_t));
         long long *overhead_nsec = (long long*)malloc(nprofiles * sizeof(long long));

         MPI_Status status;
         PMPI_Recv (gids, nprofiles, MPI_INT, irank, irank, MPI_COMM_WORLD, &status);
         PMPI_Recv (cbids, nprofiles, MPI_INT, irank, irank, MPI_COMM_WORLD, &status);
         PMPI_Recv (n_calls, nprofiles, MPI_INT, irank, irank, MPI_COMM_WORLD, &status);
         PMPI_Recv (t_ms, nprofiles, MPI_FLOAT, irank, irank, MPI_COMM_WORLD, &status);
         PMPI_Recv (memcpy_bytes_1, nprofiles, MPI_LONG, irank, irank, MPI_COMM_WORLD, &status);
         PMPI_Recv (memcpy_bytes_2, nprofiles, MPI_LONG, irank, irank, MPI_COMM_WORLD, &status);
         PMPI_Recv (overhead_nsec, nprofiles, MPI_LONG_LONG, irank, irank, MPI_COMM_WORLD, &status);
         for (int iprof = 0; iprof < nprofiles; iprof++) {
            int gid = gids[iprof]; 
            collated_stack_t *collstack = collstacktree_ptr->stacks + gid;
            collated_cudaprofile_t *cudapapiprof = &(collstack->profile.cudaprof);

            collpapiprof->cbid = cbids[iprof];
            collpapiprof->n_calls = n_calls[iprof];
            collpapiprof->t_ms = t_ms[iprof];
            collpapiprof->memcpy_bytes[0] = memcpy_bytes_1[iprof];
            collpapiprof->memcpy_bytes[1] = memcpy_bytes_2[iprof];
            collpapiprof->overhead_nsec = overhead_nsec[iprof];
         }
         free(gids);
         free(cbids);
         free(n_calls);
         free(t_ms);
         free(memcpy_bytes_1);
         free(memcpy_bytes_2);
         free(overhead_nsec);
      }
   }
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
