#ifndef STACKS_H
#define STACKS_H

#include <stdbool.h>
#ifdef _DEBUG
#include <stdio.h>
#endif

#include "custom_types.h"
#include "symbol_types.h"
#include "stack_types.h"

int vftr_new_stack(int callerID, stacktree_t *stacktree_ptr,
                   symboltable_t symboltable, stack_kind_t stack_kind,
                   uintptr_t address, bool precise);

stacktree_t vftr_new_stacktree();

void vftr_stacktree_free(stacktree_t *stacktree_ptr);

void vftr_finalize_stacktree(stacktree_t *stacktree_ptr);

void vftr_print_stacktree(FILE *fp, stacktree_t stacktree);

void vftr_print_stacklist(FILE *fp, stacktree_t stacktree);

#endif
