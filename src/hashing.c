#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

#include <string.h>

#include "stack_types.h"
#include "sorting.h"

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
uint32_t vftr_murmur_32_scramble(uint32_t k) {
    k *= 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    return k;
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
      h = (h << 13) | (h >> 19);
      h = h * 5 + 0xe6546b64;
   }
   /* Read the rest. */
   k = 0;
   for (size_t i = length & 3; i>0; i--) {
      k <<= 8;
      k |= key[i - 1];
   }
   h ^= vftr_murmur_32_scramble(k);
   h ^= length;
   h ^= h >> 16;
   h *= 0x85ebca6b;
   h ^= h >> 13;
   h *= 0xc2b2ae35;
   h ^= h >> 16;
   return h;
}

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

void vftr_compute_stack_hashes(int nstacks, stack_t *stacks) {
   int bufferlen = 128;
   char *buffer = (char*) malloc(bufferlen*sizeof(char));
   for (int istack=0; istack<nstacks; istack++) {
      // first figure out the length of the stack string
      int stackstr_len = 0;
      int jstack = istack;
      while (jstack >= 0) {
         stackstr_len += strlen(stacks[jstack].name);
         stackstr_len += 1; // function seperator
         jstack = stacks[jstack].caller;
      }

      // realloc buffer if necessary
      if (stackstr_len > bufferlen) {
         bufferlen = stackstr_len;
         buffer = (char*) realloc(buffer, bufferlen*sizeof(char));
      }

      // copy the strings into the string buffer
      jstack = istack;
      char *ptr = buffer;
      while (jstack >= 0) {
         strcpy(ptr, stacks[jstack].name);
         ptr += strlen(stacks[jstack].name);
         *ptr = '<';
         ptr++;
         jstack = stacks[jstack].caller;
      }
      ptr--;
      *ptr = '\0';

      // compute the hash
      stacks[istack].hash =
         vftr_jenkins_murmur_64_hash(stackstr_len, (uint8_t*) buffer);
   }
   free(buffer);
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


