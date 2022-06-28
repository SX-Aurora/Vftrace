#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "sort_utils.h"

#define MAKE_SORT_NAME(x) vftr_sort_ ## x
#define SORT_NAME(typestr) MAKE_SORT_NAME(typestr)

#define MAKE_SORT_ASCENDING_NAME(x) vftr_sort_ascending ## x
#define SORT_ASCENDING_NAME(typestr) MAKE_SORT_ASCENDING_NAME(typestr)

#define MAKE_SORT_DESCENDING_NAME(x) vftr_sort_descending_ ## x
#define SORT_DESCENDING_NAME(typestr) MAKE_SORT_DESCENDING_NAME(typestr)

void SORT_ASCENDING_NAME(TYPESTR)(int n, TYPE *list) {
   if (n < 2) return;
   TYPE pivot = list[n/2];
   int left, right;
   for (left=0, right=n-1; ; left++, right--) {
      while (list[left] < pivot) left++;
      while (list[right] > pivot) right--;
      if (left >= right) break;
      TYPE temp = list[left];
      list[left] = list[right];
      list[right] = temp;
   }

   SORT_ASCENDING_NAME(TYPESTR)(left, list);
   SORT_ASCENDING_NAME(TYPESTR)(n-left, list+left);
}

void SORT_DESCENDING_NAME(TYPESTR)(int n, TYPE *list) {
   if (n < 2) return;
   TYPE pivot = list[n/2];
   int left, right;
   for (left=0, right=n-1; ; left++, right--) {
      while (list[left] > pivot) left++;
      while (list[right] < pivot) right--;
      if (left >= right) break;
      TYPE temp = list[left];
      list[left] = list[right];
      list[right] = temp;
   }

   SORT_DESCENDING_NAME(TYPESTR)(left, list);
   SORT_DESCENDING_NAME(TYPESTR)(n-left, list+left);
}


void SORT_NAME(TYPESTR)(int n, TYPE *list, bool ascending) {
   if (ascending) {
      SORT_ASCENDING_NAME(TYPESTR)(n, list);
   } else {
      SORT_DESCENDING_NAME(TYPESTR)(n, list);
   }
}

#define MAKE_SORT_PERM_NAME(x) vftr_sort_perm_ ## x
#define SORT_PERM_NAME(typestr) MAKE_SORT_PERM_NAME(typestr)

#define MAKE_SORT_PERM_ASCENDING_NAME(x) vftr_sort_perm_ascending ## x
#define SORT_PERM_ASCENDING_NAME(typestr) MAKE_SORT_PERM_ASCENDING_NAME(typestr)

#define MAKE_SORT_PERM_DESCENDING_NAME(x) vftr_sort_perm_descending_ ## x
#define SORT_PERM_DESCENDING_NAME(typestr) MAKE_SORT_PERM_DESCENDING_NAME(typestr)

void SORT_PERM_ASCENDING_NAME(TYPESTR)(int n, TYPE *list, int *perm) {
   if (n < 2) return;
   TYPE pivot = list[n/2];
   int left, right;
   for (left=0, right=n-1; ; left++, right--) {
      while (list[left] < pivot) left++;
      while (list[right] > pivot) right--;
      if (left >= right) break;
      TYPE temp = list[left];
      list[left] = list[right];
      list[right] = temp;
      // repeat swapping on permutation
      int tempi = perm[left];
      perm[left] = perm[right];
      perm[right] = tempi;
   }

   SORT_PERM_ASCENDING_NAME(TYPESTR)(left, list, perm);
   SORT_PERM_ASCENDING_NAME(TYPESTR)(n-left, list+left, perm+left);
}

void SORT_PERM_DESCENDING_NAME(TYPESTR)(int n, TYPE *list, int *perm) {
   if (n < 2) return;
   TYPE pivot = list[n/2];
   int left, right;
   for (left=0, right=n-1; ; left++, right--) {
      while (list[left] > pivot) left++;
      while (list[right] < pivot) right--;
      if (left >= right) break;
      TYPE temp = list[left];
      list[left] = list[right];
      list[right] = temp;
      // repeat swapping on permutation
      int tempi = perm[left];
      perm[left] = perm[right];
      perm[right] = tempi;
   }

   SORT_PERM_DESCENDING_NAME(TYPESTR)(left, list, perm);
   SORT_PERM_DESCENDING_NAME(TYPESTR)(n-left, list+left, perm+left);
}


void SORT_PERM_NAME(TYPESTR)(int n, TYPE *list, int **perm_ptr, bool ascending) {
   // create a unity permutation to record sorting process
   int *perm = vftr_create_unityperm(n);
   if (ascending) {
      SORT_PERM_ASCENDING_NAME(TYPESTR)(n, list, perm);
   } else {
      SORT_PERM_DESCENDING_NAME(TYPESTR)(n, list, perm);
   }
   *perm_ptr = perm;
}
