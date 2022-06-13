#include <stdlib.h>

int *vftr_create_unityperm(int n) {
   int *unityperm = (int*) malloc(n*sizeof(int));
   for (int i=0; i<n; i++) {
      unityperm[i] = i;
   }
   return unityperm;
}
