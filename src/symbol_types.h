#ifndef SYMBOL_TYPES_H
#define SYMBOL_TYPES_H

#include <stdint.h>
#include <sys/types.h>

#include "custom_types.h"

typedef struct {
   uintptr_t addr;
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

#endif
