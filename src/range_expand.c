#include <stdlib.h>
#include <stdbool.h>

#include <ctype.h>
#include <string.h>

#include "sorting.h"

static void vftr_remove_duplicates(int *n, int *list) {
   int i = 0;
   int j = 0;
   while (i < *n) {
      list[j] = list[i];
      i++;
      while (i<*n && list[i] == list[j]) {
         i++;
      }
      j++;
   }
   *n = j;
}

int vftr_char_count_in_string(char c, char *string) {
   int count = 0;
   while (*string != '\0') {
      count += *string == c;
      string++;
   }
   return count;
}

static bool vftr_is_valid_range(char *range) {
   bool valid = true;
   valid = valid && isdigit(*range);
   while (isdigit(*range)) {
      range++;
   }
   valid = *range == '-';
   range++;
   while (*range != '\0') {
      valid = valid && isdigit(*range);
      range++;
   }
   return valid;
}

static bool vftr_is_valid_number(char *string) {
   bool valid = true;
   valid = valid && isdigit(*string);
   while (*string != '\0') {
      valid = valid && isdigit(*string);
      string++;
   }
   return valid;
}

void vftr_start_endval_in_range(char *orange, int *startval, int *endval) {
   const char *hyphen = "-";
   char *range = strdup(orange);
   char *token = NULL;
   token = strtok(range, hyphen);
   *startval = atoi(token);
   token = strtok(NULL, hyphen);
   *endval = atoi(token);
   free(range);
}

int *vftr_expand_rangelist(char *olist, int *nvals_ptr) {
   if (strlen(olist) == 0) {
      *nvals_ptr = 0;
      return NULL;
   }
   const char *komma = ",";

   char *list = strdup(olist);
   int ntokens = 1 + vftr_char_count_in_string(*komma, olist);
   char **tokens = (char**) malloc(ntokens*sizeof(char*));
   // get first token
   tokens[0] = strtok(list, komma);
   // walk through other tokens
   for (int itoken=1; itoken<ntokens; itoken++) {
      tokens[itoken] = strtok(NULL, komma);
   }   
   int ninlist = 0;
   for (int itoken=0; itoken<ntokens; itoken++) {
      if (vftr_is_valid_number(tokens[itoken])) {
         ninlist++;
      } else if (vftr_is_valid_range(tokens[itoken])) {
         int startval;
         int endval;
         vftr_start_endval_in_range(tokens[itoken], &startval, &endval);
         ninlist += endval - startval + 1;
      } else {
         free(list);
         free(tokens);
         *nvals_ptr = 0;
         return NULL;
      }
   }   

   int *exp_list = (int*) malloc(ninlist*sizeof(int));
   int ival = 0;
   for (int itoken=0; itoken<ntokens; itoken++) {
      if (vftr_is_valid_number(tokens[itoken])) {
         exp_list[ival] = atoi(tokens[itoken]);
         ival++;
      } else if (vftr_is_valid_range(tokens[itoken])) {
         int startval;
         int endval;
         vftr_start_endval_in_range(tokens[itoken], &startval, &endval);
         for (int jval=startval; jval<=endval; jval++) {
            exp_list[ival] = jval;
            ival++;
         }
      }   
   }   

   free(list);
   free(tokens);

   vftr_sort_int(ninlist, exp_list, true);
   vftr_remove_duplicates(&ninlist, exp_list);

   *nvals_ptr = ninlist;
   return exp_list;
}
