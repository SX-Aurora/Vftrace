/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _MPI
#include <mpi.h>
#endif

#include "vftr_setup.h"
#include "vftr_sorting.h"

// define the sub-hash functions here as they do not need to be known
// to other functions through the header
uint32_t vftr_jenkins_32_hash(size_t length, const uint8_t* key);
uint32_t vftr_murmur_32_scramble(uint32_t k);
uint32_t vftr_murmur3_32_hash(size_t length, const uint8_t* key);

// combined hash-function
uint64_t vftr_jenkins_murmur_64_hash(size_t length, const uint8_t* key) {
   // create a 64 bit hash by concatenate the jenkins and murmur3 32 bit hashes
   // Probability of both having a collision on the same input is hopefully small
   uint32_t jenkins = vftr_jenkins_32_hash(length, key);
   uint32_t murmur3 = vftr_murmur3_32_hash(length, key);
   // jenkins is stored in the first 32 bits
   // murmur3 is stored in the last 32 bits
   uint64_t hash = ((uint64_t)jenkins << 32) | murmur3;

   return hash;
}

// jenkins 32bit hash implementation
// Taken unchanged from wikipedia 
//    (https://en.wikipedia.org/wiki/Jenkins_hash_function)
// Published on wikipedia under the creative commons licence 
//    (https://creativecommons.org/licenses/by-sa/3.0/)
uint32_t vftr_jenkins_32_hash(size_t length, const uint8_t* key) {
  uint32_t hash = 0;
  for (size_t i=0; i < length; i++) {
    hash += key[i];
    hash += hash << 10;
    hash ^= hash >> 6;
  }
  hash += hash << 3;
  hash ^= hash >> 11;
  hash += hash << 15;
  return hash;
}

// murmur3 32bit hash implementation
// Adapted from wikipedia 
//    (https://en.wikipedia.org/wiki/MurmurHash)
//    The seed of the hash function was removed and replaced 
//    by a randomly selected constant.
// Published on wikipedia under the creative commons licence 
//    (https://creativecommons.org/licenses/by-sa/3.0/)
//static inline uint32_t vftr_murmur_32_scramble(uint32_t k) {
uint32_t vftr_murmur_32_scramble(uint32_t k) {
    k *= 0xcc9e2d51; 
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    return k;
}

void vftr_murmur_32_scramble_2 (uint32_t *k) {
	*k = *k * 0xcc9e2d51;
   	*k = (*k << 15) | (*k >> 17);
	*k = *k * 0x1b873593;
}

uint32_t vftr_murmur3_32_hash(size_t length, const uint8_t* key) {
   // use an arbitrary seed;
   uint32_t h = 0x5c0940e9;
   uint32_t k;
   // Read in groups of 4
   for (size_t i = length >> 2; i>0; i--) {
      memcpy(&k, key, sizeof(uint32_t));
      key += sizeof(uint32_t);
      h ^= vftr_murmur_32_scramble(k);
      ///vftr_murmur_32_scramble_2(&k);
      ///h ^= k;
      h = (h << 13) | (h >> 19);
      h = h * 5 + 0xe6546b64;
   }
   /* Read the rest. */
   k = 0;
   for (size_t i = length & 3; i>0; i--) {
      k <<= 8;
      k |= key[i - 1];
   }
   ///vftr_murmur_32_scramble_2(&k);
   ///h ^= k;
   h ^= vftr_murmur_32_scramble(k);
   h ^= length;
   h ^= h >> 16;
   h *= 0x85ebca6b;
   h ^= h >> 13;
   h *= 0xc2b2ae35;
   h ^= h >> 16;
   return h;
}

void vftr_remove_multiple_hashes(int *n, uint64_t *hashlist) {
   // first sort the list
   vftr_radixsort_uint64(*n, hashlist);

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

   return;
}

void vftr_synchronise_hashes(int *nptr, uint64_t **hashlistptr) {
#ifdef _MPI

   // define a 64bit mpi-type
   // C does not require a long long to be 64bit. 
   // It only requires it to be at least 64bit.
   // Therefore, own MPI-type to properly communicat 64bit numbers
   MPI_Datatype mpi_uint64;
   PMPI_Type_contiguous(8, MPI_BYTE, &mpi_uint64);
   PMPI_Type_commit(&mpi_uint64);

   // get the length of each ranks hash-list
   int *lengths = NULL;
   if (vftr_mpirank == 0) {
      lengths = (int*) malloc(vftr_mpisize*sizeof(int));
   }
   PMPI_Gather(nptr, // send buffer
               1, // send count
               MPI_INT, // send type
               lengths, // receive buffer
               1, // receive count per rank
               MPI_INT, // receive type
               0, // root process
               MPI_COMM_WORLD); // communicator

   // get the sum of all length
   int totallength = 0;
   if (vftr_mpirank == 0) {
      for (int irank=0; irank<vftr_mpisize; irank++) {
         totallength += lengths[irank];
      }

   }

   // allocate buffer for all hashlists
   // and construct displacement vector
   uint64_t *totalhashlist = NULL;
   int *displs = NULL;
   if (vftr_mpirank == 0) {
      totalhashlist = (uint64_t*) malloc(totallength*sizeof(uint64_t));
      displs = (int*) malloc(vftr_mpisize*sizeof(int));
      displs[0] = 0;
      for (int irank=1; irank<vftr_mpisize; irank++) {
         displs[irank] = displs[irank-1] + lengths[irank-1];
      }
   }

   // Gather the list of hashes from every rank into one list
   PMPI_Gatherv(*hashlistptr, // send buffer
                *nptr, // send count
                mpi_uint64, // send type
                totalhashlist, // receive buffer
                lengths, // receive count vector
                displs, // receive displacement vector
                mpi_uint64, // receive type
                0, // root process
                MPI_COMM_WORLD); // communicator

   // clean hashlist of multiple entries
   if (vftr_mpirank == 0) {
      vftr_remove_multiple_hashes(&totallength, totalhashlist);
   }

   // distribute the new length to all ranks
   PMPI_Bcast(&totallength, // send/receive buffer
              1, // send/receive count
              MPI_INT, // send/receive type
              0, // root process
              MPI_COMM_WORLD); // communicator

   // If the length has changed reallocate the local hash-list size
   // (Hash list size can only grow)
   if (totallength != *nptr) {
      free(*hashlistptr);
      *hashlistptr = (uint64_t*) malloc(totallength*sizeof(uint64_t));
      *nptr = totallength;
   }

   // rank 0 copies the total hashlist to the hashlist
   // this makes the Bcast command easier
   if (vftr_mpirank == 0) {
      for (int i=0; i<*nptr; i++) {
         (*hashlistptr)[i] = totalhashlist[i];
      }
   }

   // distribute the hashlist to all ranks
   PMPI_Bcast(*hashlistptr, // send/receive buffer
              totallength, // send/receive count
              mpi_uint64, // send/receive type
              0, // root process
              MPI_COMM_WORLD); // communicator

   // free the local resources
   // The custom MPI-type
   MPI_Type_free(&mpi_uint64);

   // The rank 0 exclusive arrays for the MPI-communication
   if (vftr_mpirank == 0) {
      free(lengths);
      lengths = NULL;

      free(displs);
      displs = NULL;

      free(totalhashlist);
      displs = NULL;
   }

#endif
   return;
}






