#ifndef FUNCTION_TYPES_H
#define FUNCTION_TYPES_H

typedef struct {
   char *name;
   int id;
} function_t;

typedef struct {
   int nfunctions;
   int maxfunctions;
   function_t *functions;
} functionlist_t;

#endif
