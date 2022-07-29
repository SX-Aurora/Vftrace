#ifndef SYMBOLS_H
#define SYMBOLS_H

#include <stdio.h>
#include <stdbool.h>

#include "symbol_types.h"
#include "environment_types.h"

symboltable_t vftr_read_symbols();

void vftr_symboltable_free(symboltable_t *symboltable_ptr);

void vftr_print_symbol_table(FILE *fp, symboltable_t symboltable);

void vftr_symboltable_determine_preciseness(symboltable_t *symboltable_ptr,
                                            regex_t *preciseregex);

void vftr_symboltable_strip_fortran_module_name(symboltable_t *symboltable_ptr,
                                                bool strip_module_names);
void vftr_symboltable_demangle_cxx_name(symboltable_t *symboltable_ptr,
                                        bool demangle_cxx);

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
