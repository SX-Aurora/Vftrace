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
#include <stdlib.h>

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
   return;
}

/**********************************************************************/

int vftr_compare_integer_ascending (const void *a1, const void *a2) {
   return (*(int*)a1 - *(int*)a2);
}

int vftr_compare_integer_descending (const void *a1, const void *a2) {
   return (*(int*)a2 - *(int*)a1);
}

int vftr_compare_double_ascending (const void *a1, const void *a2) {
   return (*(double*)a1 - *(double*)a2);
}

int vftr_compare_double_descending (const void *a1, const void *a2) {
   return (*(double*)a2 - *(double*)a1);
}

void vftr_sort_integer (int **i_array, int n, bool ascending) {
   int *tmp = (int*)malloc (n * sizeof(int));
   for (int i = 0; i < n; i++) {
      tmp[i] = (*i_array)[i];
   }

   if (ascending) {
      qsort (tmp, (size_t)n, sizeof(int), vftr_compare_integer_ascending);
   } else {
      qsort (tmp, (size_t)n, sizeof(int), vftr_compare_integer_descending);
   }

   for (int i = 0; i < n; i++) {
      (*i_array)[i] = tmp[i];
   }
   free(tmp);
}

void vftr_sort_double (double **d_array, int n, bool ascending) {
   double *tmp = (double*) malloc (n * sizeof(double)); 
   for (int i = 0; i < n; i++) {
      tmp[i] = (*d_array)[i];
   }

   if (ascending) {
      qsort (d_array, (size_t)n, sizeof(int*), vftr_compare_double_ascending);
   } else {
      qsort (d_array, (size_t)n, sizeof(int*), vftr_compare_double_descending);
   }

   for (int i = 0; i < n; i++) {
      (*d_array)[i] = tmp[i];
   }
   free(tmp);
}

typedef struct index_container_double {
  double value;
  int index;
} index_container_double_t; 

int vftr_compare_index_container (const void *a1, const void *a2) {
   index_container_double_t c1 = *(index_container_double_t *)a1; 
   index_container_double_t c2 = *(index_container_double_t *)a2; 
   double diff = c2.value - c1.value;
   if (diff > 0) return 1;
   if (diff < 0) return -1;
   return 0;
}


void vftr_sort_double_with_indices (double **values, int **indices, int n) {
  index_container_double_t c[n];
  for (int i = 0; i < n; i++) {
     c[i].value = (*values)[i];
     c[i].index = i;
  }   
  qsort (c, (size_t)n, sizeof(index_container_double_t), vftr_compare_index_container);
  for (int i = 0; i < n; i++) { 	
     (*values)[i] = c[i].value;
     (*indices)[i] = c[i].index;
  }
}
 

