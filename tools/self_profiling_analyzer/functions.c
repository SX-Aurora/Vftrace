#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "realloc_consts.h"
#include "function_types.h"

function_t new_function(char *name) {
   function_t function;
   function.name = strdup(name);
   return function;
}

function_t first_function() {
   return new_function("vftrace");
}

void free_function(function_t *function_ptr) {
   if (function_ptr->name != NULL) {
      free(function_ptr->name);
      function_ptr->name = NULL;
      function_ptr->id = -1;
   }
}

void print_function(FILE *fp, function_t function) {
   fprintf(fp, "%d: %s\n",
           function.id,
           function.name);
}

void functionlist_realloc(functionlist_t *functionlist_ptr) {
   functionlist_t functionlist = *functionlist_ptr;
   while (functionlist.nfunctions > functionlist.maxfunctions) {
      int maxfunctions = functionlist.maxfunctions*realloc_rate+realloc_add;
      functionlist.functions = (function_t*)
         realloc(functionlist.functions, maxfunctions*sizeof(function_t));
      functionlist.maxfunctions = maxfunctions;
   }
   *functionlist_ptr = functionlist;
}

void append_function_to_functionlist(function_t function,
                                     functionlist_t *functionlist_ptr) {
   int idx = functionlist_ptr->nfunctions;
   functionlist_ptr->nfunctions++;
   functionlist_realloc(functionlist_ptr);
   function.id = idx;
   functionlist_ptr->functions[idx] = function;
}

int function_index_from_functionlist_by_name(char *name,
                                             functionlist_t *functionlist_ptr) {
   int idx = -1;
   for (int ifunc=0; ifunc<functionlist_ptr->nfunctions; ifunc++) {
      function_t *function_ptr = functionlist_ptr->functions+ifunc;
      if (strcmp(name, function_ptr->name) == 0) {
         idx = ifunc;
         break;
      }
   }

   if (idx == -1) {
      idx = functionlist_ptr->nfunctions;
      append_function_to_functionlist(new_function(name), functionlist_ptr);
   }

   return idx;
}

functionlist_t new_functionlist() {
   functionlist_t functionlist;
   functionlist.maxfunctions = 0;
   functionlist.nfunctions = 0;
   functionlist.functions = NULL;
   append_function_to_functionlist(first_function(), &functionlist);
   return functionlist;
}

void free_functionlist(functionlist_t *functionlist_ptr) {
   if (functionlist_ptr->maxfunctions > 0) {
      for (int ifunc=0; ifunc<functionlist_ptr->nfunctions; ifunc++) {
         free_function(functionlist_ptr->functions+ifunc);
      }
      free(functionlist_ptr->functions);
      functionlist_ptr->maxfunctions = 0;
      functionlist_ptr->nfunctions = 0;
      functionlist_ptr->functions = NULL;
   }
}

void print_functionlist(FILE *fp, functionlist_t functionlist) {
   for (int ifunc=0; ifunc<functionlist.nfunctions; ifunc++) {
      print_function(fp, functionlist.functions[ifunc]);
   }
}
