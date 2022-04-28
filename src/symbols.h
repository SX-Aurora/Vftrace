#ifndef SYMBOLS_H
#define SYMBOLS_H

#include <stdio.h>

#include "symbol_types.h"

symboltable_t vftr_read_symbols();

void vftr_symboltable_free(symboltable_t *symboltable_ptr);

void vftr_print_symbol_table(FILE *fp, symboltable_t symboltable);

char *vftr_get_name_from_address(symboltable_t symboltable,
                                 uintptr_t address);

#endif
