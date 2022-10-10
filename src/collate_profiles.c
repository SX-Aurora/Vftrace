#include <stdlib.h>

#ifdef _MPI
#include <mpi.h>
#endif

#include "self_profile.h"
#include "stack_types.h"
#include "collated_stack_types.h"
#include "collate_callprofiles.h"
#ifdef _MPI
#include "collate_mpiprofiles.h"
#endif
#ifdef _OMP
#include "collate_ompprofiles.h"
#endif
#ifdef _CUPTI
#include "collate_cuptiprofiles.h"
#endif

void vftr_collate_profiles(collated_stacktree_t *collstacktree_ptr,
                           stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   int myrank;
   int nranks;
   int nprofiles = stacktree_ptr->nstacks;
   int *nremote_profiles = NULL;
#ifdef _MPI
   PMPI_Comm_rank(MPI_COMM_WORLD, &myrank);
   PMPI_Comm_size(MPI_COMM_WORLD, &nranks);
#else
   myrank = 0;
   nranks = 1;
#endif

   // get the number of profiles to collect from each rank
   if (myrank == 0) {
      nremote_profiles = (int*) malloc(nranks*sizeof(int));
   }
#ifdef _MPI
   PMPI_Gather(&nprofiles, 1,
               MPI_INT,
               nremote_profiles, 1,
               MPI_INT,
               0, MPI_COMM_WORLD);
#else
   nremote_profiles[0] = nprofiles;
#endif

   vftr_collate_callprofiles(collstacktree_ptr, stacktree_ptr,
                             myrank, nranks, nremote_profiles);
#ifdef _MPI
   vftr_collate_mpiprofiles(collstacktree_ptr, stacktree_ptr,
                            myrank, nranks, nremote_profiles);
#endif

#ifdef _OMP
   vftr_collate_ompprofiles(collstacktree_ptr, stacktree_ptr,
                            myrank, nranks, nremote_profiles);
#endif

#ifdef _CUPTI
   vftr_collate_cuptiprofiles(collstacktree_ptr, stacktree_ptr,
                              myrank, nranks, nremote_profiles);
#endif

   free(nremote_profiles);
   SELF_PROFILE_END_FUNCTION;
}
