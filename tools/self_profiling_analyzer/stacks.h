#ifndef STACKS_H
#define STACKS_H

#include <stdlib.h>
#include <stdio.h>

#include "function_types.h"
#include "stack_types.h"

int search_callee(vftr_stack_t *stacks, int callerID, char *name);

void stack_callees_realloc(vftr_stack_t *stack_ptr);

void stacktree_realloc(stacktree_t *stacktree_ptr);

void insert_callee(int calleeID, vftr_stack_t *caller);

int new_stack(int callerID, char *name, stacktree_t *stacktree_ptr);

vftr_stack_t first_stack(char *name);

void free_stack(vftr_stack_t *stacks_ptr, int stackID);

stacktree_t new_stacktree(functionlist_t functionlist);

void free_stacktree(stacktree_t *stacktree_ptr);

void update_stacks_exclusive_time(stacktree_t *stacktree_ptr);

void finalize_stacktree(stacktree_t *stacktree_ptr);

vftr_stack_t **sort_stacks_by_excl_time(stacktree_t stacktree);

void print_stack_branch(FILE *fp, int level, stacktree_t stacktree, int stackid);

void print_stacktree(FILE *fp, stacktree_t stacktree);

int get_stack_string_length(stacktree_t stacktree, int stackid);

char *get_stack_string(stacktree_t stacktree, int stackid);

void print_stack(FILE *fp, stacktree_t stacktree, int stackid);

void print_stacklist(FILE *fp, stacktree_t stacktree);

void print_sorted_stacklist(FILE *fp, vftr_stack_t **sortedstacklist,
                            stacktree_t stacktree);

#endif
