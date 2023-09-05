#ifndef COLLATED_STACK_TYPES_H
#define COLLATED_STACK_TYPES_H

#include <stdint.h>

#include <stdbool.h>

#include "stack_types.h"
#include "collated_profiling_types.h"

typedef struct {
   int ngids;
   int maxgids;
   int *gids;
} gid_list_t;

typedef struct {
   // local stack
   vftr_stack_t *local_stack;
   // profiling data for collective logfile
   collated_profile_t profile;
   // gloabl stackID
   int gid;
   // this list is only needed for grouped stacks
   gid_list_t gid_list;
   bool precise;
   int caller;
   int ncallees;
   int *callees;
   // need a copy of the name for even for local functions
   // because functions from other processes might
   // not appear in the local symbol table (e.g. two binaries)
   char *name;
   // hash
   uint64_t hash;
} collated_stack_t;

typedef struct {
   int nstacks;
   int maxstacks;
   collated_stack_t *stacks;
   // whether this is a name grouped stack or regular
   bool namegrouped;
} collated_stacktree_t;

#endif
