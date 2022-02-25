#ifndef SYMBOLS_H
#define SYMBOLS_H

#include <sys/types.h>

typedef struct {
   unsigned long long addr;
   char *name; /* Not de-mangled function name */
   int index; /* Section index */
} symbol_t;

typedef struct {
   unsigned int nsymbols;
   symbol_t *symbols;
} symboltable_t;

// types to contain library paths
typedef struct {
   off_t base;
   off_t offset;
   char *path;
} library_t;

typedef struct {
   int nlibraries;
   int maxlibraries;
   library_t *libraries;
} librarylist_t;

symboltable_t vftr_read_symbols();

void vftr_symboltable_free(symboltable_t *symboltable_ptr);

#endif
