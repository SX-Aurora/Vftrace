#ifndef SYMBOLS_H
#define SYMBOLS_H

#include "symbol_types.h"

symboltable_t vftr_read_symbols();

void vftr_symboltable_free(symboltable_t *symboltable_ptr);

void vftr_print_symbol_table(FILE *fp, symboltable_t symboltable);

#endif
