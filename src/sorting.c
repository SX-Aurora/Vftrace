#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "custom_types.h"
#include "symbols.h"

// sorts a list of unsigned 64 bit integer with linear scaling radix sort
// one bit at a time
void vftr_radixsort_uint64(int n, uint64_t *list) {
   // create the buckets for sorting
   // use one array to store both buckets
   uint64_t *buckets = (uint64_t*) malloc(2*n*sizeof(uint64_t));
   // loop over the bits
   int nbits = 8*sizeof(uint64_t);
   for (int ibit=0; ibit<nbits; ibit++) {
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
   free(buckets);
}

// sorts a list of integers with linear scaling radix sort
// one bit at a time
void vftr_radixsort_int(int n, int *list) {
   // create the buckets for sorting
   // use one array to store both buckets
   int *buckets = (int*) malloc(2*n*sizeof(int));
   // loop over the bits
   int nbits = 8*sizeof(int);
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

void vftr_sort_integer(int *list, int n, bool ascending) {
   vftr_radixsort_int(n, list);
   if (!ascending) {
      // revert array order
      for (int i=0; i<n/2; i++) {
         int tmpint = list[i];
         list[i] = list[n-i-1];
         list[n-i-1] = tmpint;
      }
   }
}

void vftr_sort_double(double *list, int n, bool ascending) {
   vftr_radixsort_double(n, list);
   if (!ascending) {
      // revert array order
      for (int i=0; i<n/2; i++) {
         double tmpdbl = list[i];
         list[i] = list[n-i-1];
         list[n-i-1] = tmpdbl;
      }
   }
}

void vftr_sort_double_copy (double *d_array, int n, bool ascending, double *d_copy) {
   for (int i=0; i<n; i++) {
      d_copy[i] = d_array[i];
   }
   vftr_sort_double(d_copy, n, ascending);
}
