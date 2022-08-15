#include <stdint.h>
#include <stdlib.h>

#include "stack_types.h"
#include "collated_stack_types.h"

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

//   // Technically faster, but consumes more memory
//   TYPE *tmplist = (TYPE*) malloc(n*sizeof(TYPE));
//   for (int i=0; i<n; i++) {
//      tmplist[i] = list[perm[i]];
//   }
//   for (int i=0; i<n; i++) {
//      list[i] = tmplist[i];
//   }
//   free(tmplist);
}
