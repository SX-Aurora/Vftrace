#include <stdlib.h>

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

#define MAKE_RADIXSORT_NAME(x) vftr_radixsort_ ## x
#define RADIXSORT_NAME(typestr) MAKE_RADIXSORT_NAME(typestr)

// sorts a list of floats with linear scaling radix sort
// one bit at a time
void RADIXSORT_NAME(TYPESTR)(int n, TYPE *list) {
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
         list[i] = buckets[0*n+i];
      }
      for (int i=0; i<idx[1]; i++) {
         list[idx[0]+i] = buckets[1*n+i];
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
   for (int i=0; i<idx[1]; i++) {
      list[idx[1]-i-1] = buckets[1*n+i];
   }
   for (int i=0; i<idx[0]; i++) {
      list[idx[1]+i] = buckets[0*n+i];
   }
   free(buckets);
}

#define MAKE_RADIXSORT_PERM_NAME(x) vftr_radixsort_perm_ ## x
#define RADIXSORT_PERM_NAME(typestr) MAKE_RADIXSORT_PERM_NAME(typestr)

// sorts a list of floats with linear scaling radix sort
// one bit at a time
// Record the sorting process in a permutation for later use
void RADIXSORT_PERM_NAME(TYPESTR)(int n, TYPE *list, int **perm_ptr) {
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
   for (int i=0; i<idx[1]; i++) {
      list[idx[1]-i-1] = buckets[1*n+i];
      perm[idx[1]-i-1] = pbuckets[1*n+i];
   }
   for (int i=0; i<idx[0]; i++) {
      list[idx[1]+i] = buckets[0*n+i];
      perm[idx[1]+i] = pbuckets[0*n+i];
   }
   free(buckets);
   free(pbuckets);
   *perm_ptr = perm;
}

#define MAKE_APPLY_PERM_NAME(x) vftr_apply_perm_ ## x
#define APPLY_PERM_NAME(typestr) MAKE_APPLY_PERM_NAME(typestr)
void APPLY_PERM_NAME(TYPESTR)(int n, TYPE *list, int *perm) {
   for (int i=0; i<n; i++) {
      if (perm[i] > 0) {
         TYPE tmp = list[i];
         list[i] = list[perm[i]];
         perm[i] *= -1;
         int next = -perm[i];
         int prev = i;
         while (next != i) {
            prev = next;
            list[next] = list[perm[next]];
            perm[next] *= -1;
            next = -perm[next];
         }
         list[prev] = tmp;
      }
   }
   for (int i=0; i<n; i++) {
      perm[i] *= -1;
   }

   // Technically faster, but consumes more memory
//   TYPE *tmplist = (TYPE*) malloc(n*sizeof(TYPE));
//   for (int i=0; i<n; i++) {
//      tmplist[i] = list[perm[i]];
//   }
//   for (int i=0; i<n; i++) {
//      list[i] = tmplist[i];
//   }
//   free(tmplist);
}
