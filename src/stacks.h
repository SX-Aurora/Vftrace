#ifndef STACKS_H
#define STACKS_H

#include <stdbool.h>
#include <stdio.h>

#include "custom_types.h"
#include "symbol_types.h"
#include "stack_types.h"

int vftr_new_stack(int callerID, stacktree_t *stacktree_ptr,
                   const char *name, const char *cleanname,
                   uintptr_t address, bool precise);

stacktree_t vftr_new_stacktree();

void vftr_stacktree_free(stacktree_t *stacktree_ptr);

void vftr_finalize_stacktree(stacktree_t *stacktree_ptr);

void vftr_print_stacktree(FILE *fp, stacktree_t stacktree);

void vftr_print_stacklist(FILE *fp, stacktree_t stacktree);

int vftr_get_stack_string_length(stacktree_t stacktree, int stackid, bool show_precise);

char *vftr_get_stack_string(stacktree_t stacktree, int stackid, bool show_precise);

void vftr_stacktree_realloc(stacktree_t *stacktree_ptr);

#endif
