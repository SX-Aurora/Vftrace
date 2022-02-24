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

// types to contain paths
typedef struct {
   off_t base;
   off_t offset;
   char *path;
} path_t;

typedef struct {
   unsigned int npaths;
   unsigned int maxpaths;
   path_t *paths;
} pathlist_t;

symbollist_t vftr_read_symbols();

#endif
