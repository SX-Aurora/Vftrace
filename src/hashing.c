#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

#include <string.h>

#include "self_profile.h"
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

void vftr_compute_stack_hashes(stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   int nstacks = stacktree_ptr->nstacks;
   vftr_stack_t *stacks = stacktree_ptr->stacks;
   int bufferlen = 128;
   char *buffer = (char*) malloc(bufferlen*sizeof(char));

   // size of an address printed in hexadecimal with some decoration
   // e.g. (0x563d48737d058460) is 20 chars
   int addrstringsize = 2 * sizeof(long int) + 4;
   for (int istack = 0; istack < nstacks; istack++) {
      // first figure out the length of the stack string
      int stackstr_len = 0;
      int jstack = istack;
      while (jstack >= 0) {
         stackstr_len += strlen(stacks[jstack].name);
         // if a function with an unknown function name occours
         // include the address in the hash to avoid collisions
         if (!strcmp(stacks[jstack].name, "(UnknownFunctionName)")) {
            stackstr_len += addrstringsize;
         }
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
         // if a function with an unknown function name occurs
         // include the address in the hash to avoid collisions
         if (!strcmp(stacks[jstack].name, "(UnknownFunctionName)")) {
            snprintf(ptr, addrstringsize+1,
                     "(0x%0*lx)", (int) (2*sizeof(long int)),
                     (long int) stacks[jstack].address);
            ptr += addrstringsize;
         }
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
   SELF_PROFILE_END_FUNCTION;
}
