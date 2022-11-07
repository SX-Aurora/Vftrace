#include <stdlib.h>
#include <stdio.h>

#include <string.h>

int vftr_levenshtein_distance(char *s, char *t) {
   int m = strlen(s);
   int n = strlen(t);
   // create two work vectors of integer distances
   int *v0 = (int*) malloc((n+1)*sizeof(int));
   int *v1 = (int*) malloc((n+1)*sizeof(int));

   // initialize v0 (the previous row of distances)
   // this row is A[0][i]: edit distance from an empty s to t;
   // that distance is the number of characters to append to  s to make t.
   for (int i=0; i<n+1; i++) {
      v0[i] = i;
   }

   for (int i=0; i<m; i++) {
      // calculate v1 (current row distances) from the previous row v0

      // first element of v1 is A[i + 1][0]
      // edit distance is delete (i + 1) chars from s to match empty t
      v1[0] = i + 1;

      // use formula to fill in the rest of the row
      for (int j=0; j<n; j++) {
         // calculating costs for A[i + 1][j + 1]
         int deleteCost = v0[j+1] + 1;
         int insertCost = v1[j]   + 1;
         int substCost = s[i] == t[j] ? v0[j] : v0[j]+1;
         v1[j+1] = deleteCost;
         v1[j+1] = insertCost < v1[j+1] ? insertCost : v1[j+1];
         v1[j+1] = substCost < v1[j+1] ? substCost : v1[j+1];
      }

      // swap v1 (current row) with v0 (previous row) for next iteration
      int *vtmp = v0;
      v0 = v1;
      v1 = vtmp;
   }

   // after last swap, the results of v1 are now in v0
   int result = v0[n];
   free(v0);
   free(v1);
   return result;
}
