#ifndef DUMMY_SYMBOL_TABLE_H
#define DUMMY_SYMBOL_TABLE_H

#include "symbol_types.h"

symboltable_t dummy_symbol_table(int nsymbols, uintptr_t baseaddr);

void free_dummy_symbol_table(symboltable_t *symboltable);

#endif
