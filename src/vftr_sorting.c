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
#include <stdbool.h>

#include "vftr_functions.h"
#include "vftr_environment.h"

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
   int ia1 = *(int*)a1;
   int ia2 = *(int*)a2;
   if (ia1 < ia2) {
      return -1;
   } else if (ia1 > ia2) {
      return 1;
   } else {
      return 0;
   }
}

int vftr_compare_integer_descending (const void *a1, const void *a2) {
   int ia1 = *(int*)a1;
   int ia2 = *(int*)a2;
   if (ia1 < ia2) {
      return 1;
   } else if (ia1 > ia2) {
      return -1;
   } else {
      return 0;
   }
}

int vftr_compare_double_ascending (const void *a1, const void *a2) {
   double ia1 = *(double*)a1;
   double ia2 = *(double*)a2;
   if (ia1 < ia2) {
      return -1;
   } else if (ia1 > ia2) {
      return 1;
   } else {
      return 0;
   }
}

int vftr_compare_double_descending (const void *a1, const void *a2) {
   double ia1 = *(double*)a1;
   double ia2 = *(double*)a2;
   if (ia1 < ia2) {
      return 1;
   } else if (ia1 > ia2) {
      return -1;
   } else {
      return 0;
   }
}

void vftr_sort_integer (int *i_array, int n, bool ascending) {
   if (ascending) {
      qsort ((void*) i_array, (size_t)n, sizeof(int), vftr_compare_integer_ascending);
   } else {
      qsort ((void*) i_array, (size_t)n, sizeof(int), vftr_compare_integer_descending);
   }
}

void vftr_sort_double (double *d_array, int n, bool ascending) {
   if (ascending) {
      qsort ((void*) d_array, (size_t)n, sizeof(double), vftr_compare_double_ascending);
   } else {
      qsort ((void*) d_array, (size_t)n, sizeof(double), vftr_compare_double_descending);
   }
}

void vftr_sort_double_copy (double *d_array, int n, bool ascending, double *d_copy) {
   for (int i=0; i<n; i++) {
      d_copy[i] = d_array[i];
   }
   vftr_sort_double(d_copy, n, ascending);
}

/**********************************************************************/

int vftr_compare_function_excl_time (const void *a1, const void *a2) {
    function_t *f1 = *(function_t **)a1;
    function_t *f2 = *(function_t **)a2;
    if(!f2) return -1; /* Order important to work around SX qsort problem */
    if(!f1) return  1;
    long long t1 = f1->prof_current.timeExcl - f1->prof_previous.timeExcl;
    long long t2 = f2->prof_current.timeExcl - f2->prof_previous.timeExcl;
    long long diff = t2 - t1;
    if (diff > 0) return  1;
    if (diff < 0) return -1;
    return  0;
}

/**********************************************************************/

int vftr_compare_function_incl_time (const void *a1, const void *a2) {
    function_t *f1 = *(function_t **)a1;
    function_t *f2 = *(function_t **)a2;
    if(!f2) return -1; /* Order important to work around SX qsort problem */
    if(!f1) return  1;
    long long t1 = f1->prof_current.timeIncl - f1->prof_previous.timeIncl;
    long long t2 = f2->prof_current.timeIncl - f2->prof_previous.timeIncl;
    long long diff = t2 - t1;
    if (diff > 0) return  1;
    if (diff < 0) return -1;
    return  0;
}

/**********************************************************************/

int vftr_compare_function_n_calls (const void *a1, const void *a2) {
    function_t *f1 = *(function_t **)a1;
    function_t *f2 = *(function_t **)a2;
    if(!f2) return -1; /* Order important to work around SX qsort problem */
    if(!f1) return  1;
    long long n1 = f1->prof_current.calls - f1->prof_previous.calls;
    long long n2 = f2->prof_current.calls - f2->prof_previous.calls;
    long long diff = n2 - n1;
    if (diff > 0) return  1;
    if (diff < 0) return -1;
    return  0;
}

/**********************************************************************/

int vftr_compare_function_stack_id (const void *a1, const void *a2) {
    function_t *f1 = *(function_t **)a1;
    function_t *f2 = *(function_t **)a2;
    if(!f2) return -1; /* Order important to work around SX qsort problem */
    if(!f1) return  1;
    int diff = f2->gid - f1->gid;
    // In contrast to the above sorting functions, we want to have stack IDs in ascending order.
    if (diff > 0) return  -1;
    if (diff < 0) return 1;
    return  0;
}

/**********************************************************************/


int (*vftr_get_compare_function()) (const void *, const void *) {
  switch (vftr_profile_sorting_method()) {
    case EXCL_TIME: 
       return vftr_compare_function_excl_time;
    case INCL_TIME: 
       return vftr_compare_function_incl_time;
    case N_CALLS:
       return vftr_compare_function_n_calls;
    case STACK_ID:
       return vftr_compare_function_stack_id;
    default: 
       return vftr_compare_function_excl_time;
   }
}

/**********************************************************************/


