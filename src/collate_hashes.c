#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include "self_profile.h"
#include "stack_types.h"
#include "collated_hash_types.h"
#include "sorting.h"

#ifdef _MPI
#include <mpi.h>
#endif

void vftr_remove_multiple_hashes(int *n, uint64_t *hashlist) {
   SELF_PROFILE_START_FUNCTION;
   // first sort the list
   vftr_sort_uint64(*n, hashlist, true);

   // loop over the list
   int j = 0;
   for (int i=1; i<*n; i++) {
      // if a duplicate is encountered skip it
      if (hashlist[i] != hashlist[j]) {
         j++;
         hashlist[j] = hashlist[i];
      }
   }
   *n=j+1;
   SELF_PROFILE_END_FUNCTION;
}

hashlist_t vftr_collate_hashes(stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   hashlist_t stackhashes;
   stackhashes.nhashes = stacktree_ptr->nstacks;
   // fill a local list of stack hashes
   stackhashes.hashes = (uint64_t*) malloc(stackhashes.nhashes*sizeof(uint64_t));
   for (int istack=0; istack<stackhashes.nhashes; istack++) {
      stackhashes.hashes[istack] = stacktree_ptr->stacks[istack].hash;
   }

#ifdef _MPI
   int mpi_initialized;
   PMPI_Initialized(&mpi_initialized);
   if (mpi_initialized) {
      int nranks = 0;
      PMPI_Comm_size(MPI_COMM_WORLD, &nranks);
      int myrank = 0;
      PMPI_Comm_rank(MPI_COMM_WORLD, &myrank);
      // define a 64bit mpi-type
      // C does not require a long long to be 64bit.
      // It only requires it to be at least 64bit.
      // Therefore, own MPI-type to properly communicat 64bit numbers
      MPI_Datatype mpi_uint64;
      PMPI_Type_contiguous(8, MPI_BYTE, &mpi_uint64);
      PMPI_Type_commit(&mpi_uint64);
      // number of stacks on each rank
      int *nhashes_list = NULL;
      if (myrank == 0) {
         nhashes_list = (int*) malloc(nranks*sizeof(int));
      }
      PMPI_Gather(&(stackhashes.nhashes), // send buffer
                  1, // send count
                  MPI_INT, // send type
                  nhashes_list, // receive buffer
                  1, // receive count per rank
                  MPI_INT, // receive type
                  0, // root process
                  MPI_COMM_WORLD); // communicator

      // get the sum of the number of stacks from each process
      int ntothashes = 0;
      if (myrank == 0) {
         for (int irank=0; irank<nranks; irank++) {
            ntothashes += nhashes_list[irank];
         }
      }

      // allocate buffer for all hashlists
      // and construct displacement vector
      uint64_t *allhashes = NULL;
      int *displs = NULL;
      if (myrank == 0) {
         allhashes = (uint64_t*) malloc(ntothashes*sizeof(uint64_t));
         displs = (int*) malloc(nranks*sizeof(int));
         displs[0] = 0;
         for (int irank=1; irank<nranks; irank++) {
            displs[irank] = displs[irank-1] + nhashes_list[irank-1];
         }
      }

      // Gather the list of hashes from every rank into one list
      PMPI_Gatherv(stackhashes.hashes, // send buffer
                   stackhashes.nhashes, // send count
                   mpi_uint64, // send type
                   allhashes, // receive buffer
                   nhashes_list, // receive count vector
                   displs, // receive displacement vector
                   mpi_uint64, // receive type
                   0, // root process
                   MPI_COMM_WORLD); // communicator
      // clean hashlist of multiple entries
      if (myrank == 0) {
         vftr_remove_multiple_hashes(&ntothashes, allhashes);
      }

      // distribute the new length to all ranks
      PMPI_Bcast(&ntothashes, // send/receive buffer
                 1, // send/receive count
                 MPI_INT, // send/receive type
                 0, // root process
                 MPI_COMM_WORLD); // communicator

      // If the length has changed reallocate the local hash-list size
      // (Hash list size can only grow)
      stackhashes.hashes =
         (uint64_t*) realloc(stackhashes.hashes, ntothashes*sizeof(uint64_t));

      // rank 0 copies the total hashlist to the hashlist
      // this makes the Bcast command easier
      if (myrank == 0) {
         for (int i=0; i<ntothashes; i++) {
            stackhashes.hashes[i] = allhashes[i];
         }
      }
      stackhashes.nhashes = ntothashes;

      // distribute the hashlist to all ranks
      PMPI_Bcast(stackhashes.hashes, // send/receive buffer
                 ntothashes, // send/receive count
                 mpi_uint64, // send/receive type
                 0, // root process
                 MPI_COMM_WORLD); // communicator

      // free the local resources
      // The custom MPI-type
      PMPI_Type_free(&mpi_uint64);

      // The rank 0 exclusive arrays for the MPI-communication
      if (myrank == 0) {
         free(nhashes_list);
         free(displs);
         free(allhashes);
      }
   } else {
      vftr_sort_uint64(stackhashes.nhashes, stackhashes.hashes, true);
   }
#else
   vftr_sort_uint64(stackhashes.nhashes, stackhashes.hashes, true);
#endif

   SELF_PROFILE_END_FUNCTION;
   return stackhashes;
}

void vftr_collated_hashlist_free(hashlist_t *hashlist) {
   if (hashlist->nhashes > 0) {
      free(hashlist->hashes);
      hashlist->hashes = NULL;
      hashlist->nhashes = 0;
   }
}
