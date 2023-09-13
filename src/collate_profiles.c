#include <stdlib.h>

#ifdef _MPI
#include <mpi.h>
#endif

#include "self_profile.h"
#include "stack_types.h"
#include "collated_stack_types.h"
#include "collate_callprofiles.h"
#include "collate_hwprofiles.h"
#ifdef _MPI
#include "collate_mpiprofiles.h"
#endif
#ifdef _OMP
#include "collate_ompprofiles.h"
#endif
#ifdef _CUDA
#include "collate_cudaprofiles.h"
#endif
#ifdef _ACCPROF
#include "collate_accprofiles.h"
#endif

void vftr_collate_profiles(collated_stacktree_t *collstacktree_ptr,
                           stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   int myrank = 0;
   int nranks = 1;
   int nstacks = stacktree_ptr->nstacks;
   int *nremote_stacks = NULL;
   myrank = 0;
   nranks = 1;

#ifdef _MPI
   int mpi_initialized;
   PMPI_Initialized(&mpi_initialized);
   if (mpi_initialized) {
      PMPI_Comm_rank(MPI_COMM_WORLD, &myrank);
      PMPI_Comm_size(MPI_COMM_WORLD, &nranks);
   }
#endif

   // get the number of profiles to collect from each rank
   if (myrank == 0) {
      nremote_stacks = (int*) malloc(nranks * sizeof(int));
      nremote_stacks[0] = nstacks;
   }
#ifdef _MPI
   if (mpi_initialized) {
      PMPI_Gather(&nstacks, 1,
                  MPI_INT,
                  nremote_stacks, 1,
                  MPI_INT,
                  0, MPI_COMM_WORLD);
   }
#endif

   vftr_collate_callprofiles(collstacktree_ptr, stacktree_ptr,
                             myrank, nranks, nremote_stacks);
   vftr_collate_hwprofiles (collstacktree_ptr, stacktree_ptr,
                             myrank, nranks, nremote_stacks);
#ifdef _MPI
   vftr_collate_mpiprofiles(collstacktree_ptr, stacktree_ptr,
                            myrank, nranks, nremote_stacks);
#endif

#ifdef _OMP
   vftr_collate_ompprofiles(collstacktree_ptr, stacktree_ptr,
                            myrank, nranks, nremote_stacks);
#endif

#ifdef _CUDA
   vftr_collate_cudaprofiles(collstacktree_ptr, stacktree_ptr,
                              myrank, nranks, nremote_stacks);
#endif
#ifdef _ACCPROF
  vftr_collate_accprofiles (collstacktree_ptr, stacktree_ptr,
                            myrank, nranks, nremote_stacks);
#endif

   free(nremote_stacks);
   SELF_PROFILE_END_FUNCTION;
}
