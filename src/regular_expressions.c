#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <regex.h>

#include "vftr_regex.h"

regex_t *vftr_compile_regexp(char *pattern) {
   regex_t *r = (regex_t*) malloc(sizeof(regex_t));
   if (vftr_regcomp (r, pattern, REG_NOSUB|REG_EXTENDED)) {
       fprintf(stderr, "Vftrace: Invalid Regular Expression \"%s\"\n", pattern);
       free(r);
       r = NULL;
   }
   return r;
}

bool vftr_pattern_match(regex_t *r, char *s) {
   if (r != NULL) {
      regmatch_t regmatch[8];
      return !vftr_regexec (r, s, 0, regmatch, 0);
   } else {
      return false;
   }
}

void vftr_free_regexp (regex_t *r) {
   vftr_regfree (r);
   free(r);
}
