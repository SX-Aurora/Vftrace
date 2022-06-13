#include <stdint.h>
#include <stdlib.h>

#define MAKE_FN_NAME(x) vftr_radixsort_ ## x
#define FUNCTION_NAME(typestr) MAKE_FN_NAME(typestr)

// sorts a list of integers with linear scaling radix sort
// one bit at a time
void FUNCTION_NAME(TYPESTR)(int n, TYPE *list) {
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
         int bit = (list[i] >> ibit) & 1;
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
      int bit = (list[i] >> (nbits-1)) & 1;
      // add number to the selected bucket
      buckets[bit*n+idx[bit]] = list[i];
      idx[bit]++;
   }
   // copy the presorted numbers back to the original list
   for (int i=0; i<idx[1]; i++) {
      list[i] = buckets[1*n+i];
   }
   for (int i=0; i<idx[0]; i++) {
      list[idx[1]+i] = buckets[0*n+i];
   }
   free(buckets);
}
