#include <stdio.h>
#include <stdbool.h>

#include <string.h>

#include "misc_utils.h"

int main(int argc, char **argv) {
   (void) argc;
   (void) argv;
#define NSTRINGS 7
   char *teststrings[NSTRINGS];
   teststrings[0] = strdup("");
   teststrings[1] = strdup("String");
   teststrings[2] = strdup("A-Test_String");
   teststrings[3] = strdup("TestString_");
   teststrings[4] = strdup("_My_Teststring_");
   teststrings[5] = strdup("!Another_Teststring!!!!!!!!!!");
   teststrings[6] = strdup("..............................");

   char testchars[NSTRINGS];
   testchars[0] = '_';
   testchars[1] = '_';
   testchars[2] = '-';
   testchars[3] = '_';
   testchars[4] = '_';
   testchars[5] = '!';
   testchars[6] = '.';

   char *refstrings[NSTRINGS];
   refstrings[0] = strdup("");
   refstrings[1] = strdup("String");
   refstrings[2] = strdup("A-Test_String");
   refstrings[3] = strdup("TestString");
   refstrings[4] = strdup("_My_Teststring");
   refstrings[5] = strdup("!Another_Teststring");
   refstrings[6] = strdup("");

   bool chops_successful = true;
   for (int istring=0; istring<NSTRINGS; istring++) {
      printf("Choping trailing \"%c\" from \"%s\" -> ",
             testchars[istring],
             teststrings[istring]); fflush(stdout);
      vftr_chop_trailing_char(teststrings[istring], testchars[istring]);
      printf("\"%s\"", teststrings[istring]);
      if (strcmp(teststrings[istring], refstrings[istring]) != 0) {
         printf("(failure)\n");
         chops_successful = false;
      } else {
         printf("(success)\n");
      }

      free(teststrings[istring]);
      free(refstrings[istring]);
   }

   return chops_successful ? 0 : 1;
}
