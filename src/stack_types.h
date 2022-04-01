#ifndef STACK_TYPES_H
#define STACK_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#include "custom_types.h"
#include "profiling_types.h"

typedef enum {
   init,
   function,
   user_region,
   threaded_region
} stack_kind_t;

typedef struct {
   stack_kind_t stack_kind;
   // address of the function
   uintptr_t address;
   // is this function measured precisely?
   bool precise;
   // pointer to calling stack
   int caller;
   // pointers to called functions 
   int maxcallees;
   int ncallees;
   int *callees;
   // local stack ID
   int lid;
   // name of function on top of stack
   // only a pointer to the symbol table entry 
   // no need to deallocate
   char *name;
   // profiling data
   profile_t profiling;
   // Data that is filled in during finalization
   uint64_t hash;
} stack_t;

typedef struct {
   int nstacks;
   int maxstacks;
   stack_t *stacks;
} stacktree_t;

#endif
