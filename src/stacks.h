#ifndef STACKS_H
#define STACKS_H

#include <stdbool.h>

#include "address_type.h"
#include "symbol_types.h"
#include "stack_types.h"

int vftr_new_stack(int callerID, stacktree_t *stacktree_ptr,
                   symboltable_t symboltable, stack_kind_t stack_kind,
                   uintptr_t address, bool precise);

stacktree_t vftr_new_stacktree();

void vftr_stacktree_free(stacktree_t *stacktree_ptr);

#ifdef _DEBUG
void vftr_print_stacktree(stacktree_t stacktree);
#endif

#endif
