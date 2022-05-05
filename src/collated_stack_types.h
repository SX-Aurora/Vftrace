#ifndef COLLATED_STACK_TYPES_H
#define COLLATED_STACK_TYPES_H

#include <stdint.h>

#include <stdbool.h>

#include "stack_types.h"

typedef struct {
   // local stack
   stack_t *local_stack;
   // gloabl stackID
   int gid;
   bool precise;
   int caller;
   // need a copy of the name for even for local functions
   // because functions from other processes might
   // not appear in the local symbol table (e.g. two binaries)
   char *name;
   // hash
   uint64_t hash;
} collated_stack_t;

typedef struct {
   int nstacks;
   collated_stack_t *stacks;
} collated_stacktree_t;

#endif
