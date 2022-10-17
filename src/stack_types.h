#ifndef STACK_TYPES_H
#define STACK_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#include "custom_types.h"
#include "profiling_types.h"

typedef struct {
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
   // global stack ID
   int gid;
   // name of function on top of stack
   char *name;
   // cleaned (module striped, or demangled) name
   char *cleanname;
   // profiling data
   profilelist_t profiling;
   // Data that is filled in during finalization
   uint64_t hash;
} stack_t;

typedef struct {
   int nstacks;
   int maxstacks;
   stack_t *stacks;
} stacktree_t;

#endif
