#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <regex.h>

regex_t *vftr_compile_regexp(char *pattern) {
   regex_t *r;
   r = (regex_t*) malloc(sizeof(regex_t));
   if (regcomp(r, pattern, REG_NOSUB|REG_EXTENDED)) {
       fprintf(stderr, "Vftrace: Invalid Regular Expression \"%s\"\n", pattern);
       free(r);
       r = NULL;
   }
   return r;
}

bool vftr_pattern_match(regex_t *r, char *s) {
   if (r != NULL) {
      regmatch_t regmatch[8];
      return !regexec (r, s, 0, regmatch, 0);
   } else {
      return false;
   }
}
