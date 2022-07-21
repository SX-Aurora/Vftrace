#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <string.h>

#include "misc_utils.h"

int main(int argc, char **argv) {
   (void) argc;
   (void) argv;
#define NSTRINGS 5
   char *teststrings[NSTRINGS];
   teststrings[0] = strdup("");
   teststrings[1] = strdup("String");
   teststrings[2] = strdup("A-Test_String");
   teststrings[3] = strdup("modname_MOD_functionname");
   teststrings[4] = strdup("__modname_mod_function_with_mod_in_it");

   char *testdelimiters[NSTRINGS];
   testdelimiters[0] = "delim";
   testdelimiters[1] = "";
   testdelimiters[2] = "Test";
   testdelimiters[3] = "_MOD_";
   testdelimiters[4] = "_mod_";

   char *refstrings[NSTRINGS];
   refstrings[0] = strdup("");
   refstrings[1] = strdup("String");
   refstrings[2] = strdup("_String");
   refstrings[3] = strdup("functionname");
   refstrings[4] = strdup("function_with_mod_in_it");

   bool trims_successful = true;
   for (int istring=0; istring<NSTRINGS; istring++) {
      printf("Left trimming string from \"%s\" at \"%s\" -> ",
            teststrings[istring],
             testdelimiters[istring]); fflush(stdout);
      vftr_trim_left_with_delimiter(teststrings[istring], testdelimiters[istring]);
      printf("\"%s\"", teststrings[istring]);
      if (strcmp(teststrings[istring], refstrings[istring]) != 0) {
         printf("(failure)\n");
         trims_successful = false;
      } else {
         printf("(success)\n");
      }

      free(teststrings[istring]);
      free(refstrings[istring]);
   }

   return trims_successful ? 0 : 1;
   return 0;
}
