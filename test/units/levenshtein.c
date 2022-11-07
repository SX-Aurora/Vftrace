#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <string.h>

#include "levenshtein.h"

bool check_levenshtein(char *str1, char *str2, int target_dist) {
   int dist = vftr_levenshtein_distance(str1, str2);
   if (dist != target_dist) {
      fprintf(stdout, "levenshtein distance of \"%s\" and \"%s\" "
              "is expected to be %d, but was calculated to be %d\n",
              str1, str2, target_dist, dist);
      return false;
   } else {
      fprintf(stdout, "levenshtein distance of \"%s\" and \"%s\" "
              "is %d\n", str1, str2, dist);
      return true;
   }
}
int main(int argc, char **argv) {
   (void) argc;
   (void) argv;

   bool valid = true;

   valid = valid && check_levenshtein("test", "test", 0);
   valid = valid && check_levenshtein("test", "Test", 1);
   valid = valid && check_levenshtein("test", "tst", 1);
   valid = valid && check_levenshtein("tst", "Test", 2);
   valid = valid && check_levenshtein("tst", "Test", 2);
   valid = valid && check_levenshtein("kitten", "sitting", 3);
   valid = valid && check_levenshtein("sitting", "kitten", 3);
   valid = valid && check_levenshtein("Sunday", "Saturday", 3);
   valid = valid && check_levenshtein("Saturday", "Sunday", 3);
   valid = valid && check_levenshtein("deoxyribonucleic acid", "DNA", 21);
   valid = valid && check_levenshtein("DNA", "deoxyribonucleic acid", 21);

   return valid ? 0 : 1;
}
