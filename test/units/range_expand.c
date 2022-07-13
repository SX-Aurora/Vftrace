#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdio.h>

#include <string.h>

#include "range_expand.h"

bool cmp_int_arr(int n, int *arr, int refn, int *refarr) {
   if (n != refn) {return false;}

   for (int i=0; i<n; i++) {
      if (arr[i] != refarr[i]) {
         return false;
      }
   }
   return true;
}

void print_expansion(char *range, int n, int *arr, int refn, int *refarr) {
   printf("Rangestring: %s\n", range);
   printf("Expanded to:");
   if (arr == NULL) {
      printf(" NULL");
   } else {
      for (int i=0; i<n; i++) {
         printf(" %d", arr[i]);
      }
   }
   printf("\n");
   printf("Expected:   ");
   if (arr == NULL) {
      printf(" NULL");
   } else {
      for (int i=0; i<refn; i++) {
         printf(" %d", refarr[i]);
      }
   }
   printf("\n\n");
}

int main(int argc, char **argv) {
   (void) argc;
   (void) argv;

   int n = 0;
   int *outlist = NULL;

   char *range1 = "";
   int *reflist1 = NULL;
   int refn1 = 0;
   outlist = vftr_expand_rangelist(range1, &refn1);
   print_expansion(range1, n, outlist, refn1, reflist1);
   if (!cmp_int_arr(n, outlist, refn1, reflist1)) {
      return 1;
   }
   free(outlist);

   
   char *range2 = "2,3,5,7,11,13,17,19,23,29";
   int reflist2[] = {2,3,5,7,11,13,17,19,23,29};
   int refn2 = 10;
   outlist = vftr_expand_rangelist(range2, &n);
   print_expansion(range2, n, outlist, refn2, reflist2);
   if (!cmp_int_arr(n, outlist, refn2, reflist2)) {
      return 1;
   }
   free(outlist);

   char *range3 = "5-11";
   int reflist3[] = {5,6,7,8,9,10,11};
   int refn3 = 7;
   outlist = vftr_expand_rangelist(range3, &n);
   print_expansion(range3, n, outlist, refn3, reflist3);
   if (!cmp_int_arr(n, outlist, refn3, reflist3)) {
      return 1;
   }
   free(outlist);

   char *range4 = "5-8,2-3,17-21";
   int reflist4[] = {2,3,5,6,7,8,17,18,19,20,21};
   int refn4 = 11;
   outlist = vftr_expand_rangelist(range4, &n);
   print_expansion(range4, n, outlist, refn4, reflist4);
   if (!cmp_int_arr(n, outlist, refn4, reflist4)) {
      return 1;
   }
   free(outlist);

   char *range5 = "5-11,6-8";
   int reflist5[] = {5,6,7,8,9,10,11};
   int refn5 = 7;
   outlist = vftr_expand_rangelist(range5, &n);
   print_expansion(range5, n, outlist, refn5, reflist5);
   if (!cmp_int_arr(n, outlist, refn5, reflist5)) {
      return 1;
   }
   free(outlist);

   char *range6 = "7,3-6,9,13-14,12-17,19-20,25";
   int reflist6[] = {3,4,5,6,7,9,12,13,14,15,16,17,19,20,25};
   int refn6 = 15;
   outlist = vftr_expand_rangelist(range6, &n);
   print_expansion(range6, n, outlist, refn6, reflist6);
   if (!cmp_int_arr(n, outlist, refn6, reflist6)) {
      return 1;
   }
   free(outlist);

   return 0;
}
