#ifndef SYMBOLS_H
#define SYMBOLS_H

#include <stdio.h>

#include "symbol_types.h"
#include "environment_types.h"

symboltable_t vftr_read_symbols();

void vftr_symboltable_free(symboltable_t *symboltable_ptr);

void vftr_print_symbol_table(FILE *fp, symboltable_t symboltable);

void vftr_symboltable_determine_preciseness(symboltable_t *symboltable_ptr,
                                            regex_t *preciseregex);

int vftr_get_symbID_from_address(symboltable_t symboltable,
                                 uintptr_t address);

char *vftr_get_name_from_address(symboltable_t symboltable,
                                 uintptr_t address);

char *vftr_get_name_from_symbID(symboltable_t symboltable,
                                int symbID);

bool vftr_get_preciseness_from_address(symboltable_t symboltable,
                                       uintptr_t address);

bool vftr_get_preciseness_from_symbID(symboltable_t symboltable,
                                      int symbID);

#endif
