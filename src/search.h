#ifndef SEARCH_H
#define SEARCH_H

#include <stdint.h>

#include "custom_types.h"
#include "symbols.h"
#include "stacks.h"

int vftr_binary_search_uint64(int n, uint64_t *list, uint64_t value);

int vftr_binary_search_int(int n, int *list, int value);

int vftr_binary_search_symboltable(int nsymb, symbol_t *symbols,
                                   uintptr_t address);

int vftr_linear_search_callee(stack_t *stacks, int callerID, uintptr_t address);

#endif
