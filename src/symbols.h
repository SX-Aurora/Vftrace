#ifndef SYMBOLS_H
#define SYMBOLS_H

typedef struct {
   void *addr;
   char *name; /* Not de-mangled function name */
   int index; /* Section index */
} symbol_t;

typedef struct {
   unsigned int nsymbols;
   symbol_t *symbols;
} symbollist_t;

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

symbollist_t vftr_read_symbols();

#endif
