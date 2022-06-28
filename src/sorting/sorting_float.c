#include <stdlib.h>
#include <stdbool.h>

#include "custom_types.h"
#include "sort_utils.h"


#ifndef INTFLTTYPE
#define INTFLTTYPE uint64_t
#else
#if TYPE==float
#define INTFLTTYPE uintflt_t
#elif TYPE==double
#define INTFLTTYPE uintdbl_t
#endif
#endif

#define MAKE_RADIXSORT_NAME(x) vftr_sort_ ## x
#define RADIXSORT_NAME(typestr) MAKE_RADIXSORT_NAME(typestr)

// sorts a list of floats with linear scaling radix sort
// one bit at a time
void RADIXSORT_NAME(TYPESTR)(int n, TYPE *list, bool ascending) {
   // create the buckets for sorting
   // use one array to store both buckets
   TYPE *buckets = (TYPE*) malloc(2*n*sizeof(TYPE));
   // loop over the bits
   int nbits = 8*sizeof(TYPE);
   for (int ibit=0; ibit<nbits-1; ibit++) {
      int idx[2] = {0,0};
      // sort numbers into buckets based on their ibit-th bit
      for (int i=0; i<n; i++) {
         // extract the ibit-th bit of the i-th number
         uintdbl_t intdbl = *((uintdbl_t*)(list+i));
         int bit = (intdbl >> ibit) & 1;
         // add number to the selected bucket
         buckets[bit*n+idx[bit]] = list[i];
         idx[bit]++;
      }
      // copy the presorted numbers back to the original list
      for (int i=0; i<idx[0]; i++) {
         int bi = 0*n+i;
         list[i] = buckets[bi];
      }
      for (int i=0; i<idx[1]; i++) {
         int bi = 1*n+i;
         list[idx[0]+i] = buckets[bi];
      }
   }

   // sort by sign
   int idx[2] = {0,0};
   // sort numbers into buckets based on their nbits-th bit
   for (int i=0; i<n; i++) {
      // extract the nbits-th bit of the i-th number
      uintdbl_t intdbl = *((uintdbl_t*)(list+i));
      int bit = (intdbl >> (nbits-1)) & 1;
      // add number to the selected bucket
      buckets[bit*n+idx[bit]] = list[i];
      idx[bit]++;
   }
   // copy the presorted numbers back to the original list
   // The negative numbers are sorted inverse due
   // to their bit pattern representation (IEEE754)
   if (ascending) {
      for (int i=0; i<idx[1]; i++) {
         int bi = 1*n+idx[1]-i-1;
         list[i] = buckets[bi];
      }
      for (int i=0; i<idx[0]; i++) {
         int bi = 0*n+i;
         list[idx[1]+i] = buckets[bi];
      }
   } else {
      for (int i=0; i<idx[0]; i++) {
         int bi = 0*n+idx[0]-i-1;
         list[i] = buckets[bi];
      }
      for (int i=0; i<idx[1]; i++) {
         int bi = 1*n+i;
         list[idx[0]+i] = buckets[bi];
      }
   }
   free(buckets);
}

#define MAKE_RADIXSORT_PERM_NAME(x) vftr_sort_perm_ ## x
#define RADIXSORT_PERM_NAME(typestr) MAKE_RADIXSORT_PERM_NAME(typestr)

// sorts a list of floats with linear scaling radix sort
// one bit at a time
// Record the sorting process in a permutation for later use
void RADIXSORT_PERM_NAME(TYPESTR)(int n, TYPE *list, int **perm_ptr, bool ascending) {
   // create a unity permutation to record sorting process
   int *perm = vftr_create_unityperm(n);
   // create the buckets for sorting
   // use one array to store both buckets
   TYPE *buckets = (TYPE*) malloc(2*n*sizeof(TYPE));
   int *pbuckets = (int*) malloc(2*n*sizeof(int));
   // loop over the bits
   int nbits = 8*sizeof(TYPE);
   for (int ibit=0; ibit<nbits-1; ibit++) {
      int idx[2] = {0,0};
      // sort numbers into buckets based on their ibit-th bit
      for (int i=0; i<n; i++) {
         // extract the ibit-th bit of the i-th number
         uintdbl_t intdbl = *((uintdbl_t*)(list+i));
         int bit = (intdbl >> ibit) & 1;
         // add number to the selected bucket
         buckets[bit*n+idx[bit]] = list[i];
         pbuckets[bit*n+idx[bit]] = perm[i];
         idx[bit]++;
      }
      // copy the presorted numbers back to the original list
      for (int i=0; i<idx[0]; i++) {
         list[i] = buckets[0*n+i];
         perm[i] = pbuckets[0*n+i];
      }
      for (int i=0; i<idx[1]; i++) {
         list[idx[0]+i] = buckets[1*n+i];
         perm[idx[0]+i] = pbuckets[1*n+i];
      }
   }

   // sort by sign
   int idx[2] = {0,0};
   // sort numbers into buckets based on their nbits-th bit
   for (int i=0; i<n; i++) {
      // extract the nbits-th bit of the i-th number
      uintdbl_t intdbl = *((uintdbl_t*)(list+i));
      int bit = (intdbl >> (nbits-1)) & 1;
      // add number to the selected bucket
      buckets[bit*n+idx[bit]] = list[i];
      pbuckets[bit*n+idx[bit]] = perm[i];
      idx[bit]++;
   }
   // copy the presorted numbers back to the original list
   // The negative numbers are sorted inverse due
   // to their bit pattern representation (IEEE754)
   if (ascending) {
      for (int i=0; i<idx[1]; i++) {
         int bi = 1*n+idx[1]-i-1;
         list[i] = buckets[bi];
         perm[i] = pbuckets[bi];
      }
      for (int i=0; i<idx[0]; i++) {
         int bi = 0*n+i;
         list[idx[1]+i] = buckets[bi];
         perm[idx[1]+i] = pbuckets[bi];
      }
   } else {
      for (int i=0; i<idx[0]; i++) {
         int bi = 0*n+idx[0]-i-1;
         list[i] = buckets[bi];
         perm[i] = pbuckets[bi];
      }
      for (int i=0; i<idx[1]; i++) {
         int bi = 1*n+i;
         list[idx[0]+i] = buckets[bi];
         perm[idx[0]+i] = pbuckets[bi];
      }
   }  
   free(buckets);
   free(pbuckets);
   *perm_ptr = perm;
}
