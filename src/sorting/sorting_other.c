#include <stdint.h>
#include <stdlib.h>

#include "custom_types.h"
#include "symbols.h"

// sorts a list of doubles with linear scaling radix sort
// one bit at a time
void vftr_radixsort_double(int n, double *list) {
   // create the buckets for sorting
   // use one array to store both buckets
   double *buckets = (double*) malloc(2*n*sizeof(double));
   // loop over the bits
   int nbits = 8*sizeof(double);

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

int *vftr_create_unityperm(int n) {
   int *unityperm = (int*) malloc(n*sizeof(int));
   for (int i=0; i<n; i++) {
   }
}

// sort the symboltable with a linear scaling radixsort
void vftr_radixsort_symboltable(unsigned int nsymb, symbol_t *symbols) {
   // create the fuckets for sorting
   // use one array to store both buckets
   symbol_t *buckets = (symbol_t*) malloc(2*nsymb*sizeof(symbol_t));
   // loop over the bits of the symbols address which is (void*)
   int nbits = 8*sizeof(uintptr_t);
   for (int ibit=0; ibit<nbits; ibit++) {
      int idx[2] = {0,0};
      // sort the symbols into buckets base on thei ibit-th bit
      for (unsigned int i=0; i<nsymb; i++) {
         // extract the ibit-th bit of the i-th symbols address
         int bit = (symbols[i].addr >> ibit) & 1llu;
         // add the symbol to the selected bucket
         buckets[bit*nsymb+idx[bit]] = symbols[i];
         idx[bit]++;
      }
      // copy the presorted symbols back to the original list
      for (int i=0; i<idx[0]; i++) {
         symbols[i] = buckets[0*nsymb+i];
      }
      for (int i=0; i<idx[1]; i++) {
         symbols[idx[0]+i] = buckets[1*nsymb+i];
      }
   }
   free(buckets);
}
