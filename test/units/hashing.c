#include <stdio.h>

#include <string.h>

#include "hashing.h"

int main(int argc, char **argv) {
   (void) argc;
   (void) argv;
   char *string0 = "";
   char *string1 = "A";
   char *string2 = "When on board H.M.S. Beagle, as naturalist, ";
   char *string3 = "I was much struck with certain facts ";
   char *string4 = "in the distribution of the organic beings";
   char *loremipsum =
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
      "sed do eiusmod tempor incididunt ut labore et dolore "
      "magna aliqua. Ut enim ad minim veniam, quis nostrud "
      "exercitation ullamco laboris nisi ut aliquip ex ea commodo "
      "consequat. Duis aute irure dolor in reprehenderit in "
      "voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
      "Excepteur sint occaecat cupidatat non proident, sunt in "
      "culpa qui officia deserunt mollit anim id est laborum.";

   printf("%016lx: %s\n",
          vftr_jenkins_murmur_64_hash(strlen(string0), (uint8_t*)string0),
          string0);
   printf("%016lx: %s\n",
          vftr_jenkins_murmur_64_hash(strlen(string1), (uint8_t*)string1),
          string1);
   printf("%016lx: %s\n",
          vftr_jenkins_murmur_64_hash(strlen(string2), (uint8_t*)string2),
          string2);
   printf("%016lx: %s\n",
          vftr_jenkins_murmur_64_hash(strlen(string3), (uint8_t*)string3),
          string3);
   printf("%016lx: %s\n",
          vftr_jenkins_murmur_64_hash(strlen(string4), (uint8_t*)string4),
          string4);
   printf("%016lx: %s\n",
          vftr_jenkins_murmur_64_hash(strlen(loremipsum), (uint8_t*)loremipsum),
          loremipsum);
   return 0;
}
