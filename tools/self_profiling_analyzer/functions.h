#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stdio.h>

#include "function_types.h"

function_t new_function(char *name);

function_t first_function();

void free_function(function_t *function_ptr);

void print_function(FILE *fp, function_t function);

void functionlist_realloc(functionlist_t *functionlist_ptr);

void append_function_to_functionlist(function_t function,
                                     functionlist_t *functionlist_ptr);

int function_index_from_functionlist_by_name(char *name,
                                             functionlist_t *functionlist_ptr);

functionlist_t new_functionlist();

void free_functionlist(functionlist_t *functionlist_ptr);

void print_functionlist(FILE *fp, functionlist_t functionlist);

#endif
