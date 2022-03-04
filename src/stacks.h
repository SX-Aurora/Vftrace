#ifndef STACKS_H
#define STACKS_H

#include <stdbool.h>

#include "address_type.h"
#include "symbols.h"

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
   // local and global stack-ID
   int lid, gid;
   // name of function on top of stack
   // only a pointer to the symbol table entry 
   // no need to deallocate
   char *name;
} stack_t;

typedef struct {
   int nstacks;
   int maxstacks;
   stack_t *stacks;
} stacktree_t;

int vftr_new_stack(int callerID, stacktree_t *stacktree_ptr,
                   symboltable_t symboltable, stack_kind_t stack_kind,
                   uintptr_t address, bool precise);

stacktree_t vftr_new_stacktree();

void vftr_stacktree_free(stacktree_t *stacktree_ptr);

#ifdef _DEBUG
void vftr_print_stacktree(stacktree_t stacktree);
#endif

#endif
