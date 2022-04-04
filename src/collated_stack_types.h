#ifndef COLLATED_STACK_TYPES_H
#define COLLATED_STACK_TYPES_H

#include <stdbool.h>

#include "stack_types.h"

typedef struct {
   // local stackID
   int lid;
   // gloabl stackID
   int gid;
   int caller;
   // need a copy of the name for even for local functions
   // because functions from other processes might
   // not appear in the local symbol table (e.g. two binaries)
   char *name;
} collated_stack_t;

#endif
