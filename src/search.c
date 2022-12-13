#include <stdint.h>

#include <string.h>

#include "custom_types.h"
#include "symbols.h"
#include "stack_types.h"
#include "collated_stack_types.h"

int vftr_binary_search_uint64(int n, uint64_t *list, uint64_t value) {
   int low = 0;
   int high = n-1;
   while (low <= high) {
      int mid = (low+high) >> 1;
      if (list[mid] > value) {
         high = mid - 1;
      } else if (list[mid] < value) {
         low = mid + 1;
      } else {
         return mid;
      }
   }
   return -1; // not found
}

int vftr_binary_search_int(int n, int *list, int value) {
   int low = 0;
   int high = n-1;
   while (low <= high) {
      int mid = (low+high) >> 1;
      if (list[mid] > value) {
         high = mid - 1;
      } else if (list[mid] < value) {
         low = mid + 1;
      } else {
         return mid;
      }
   }
   return -1; // not found
}

int vftr_binary_search_symboltable(int nsymb, symbol_t *symbols,
                                   uintptr_t address) {
   int low = 0;
   int high = nsymb -1;
   while (low <= high) {
      int mid = (low+high) >> 1;
      if (symbols[mid].addr > address) {
         high = mid - 1;
      } else if (symbols[mid].addr < address) {
         low = mid + 1;
      } else {
         return mid;
      }
   }
   return -1; // not found
}

int vftr_binary_search_collated_stacks_name(collated_stacktree_t stacktree, char *name) {
   int low = 0;
   int high = stacktree.nstacks -1;
   while (low <= high) {
      int mid = (low+high) >> 1;
      char *stack_name = stacktree.stacks[mid].name;
      int compare = strcmp(stack_name, name);
      if (compare > 0) {
         high = mid -1;
      } else if (compare < 0) {
         low = mid + 1;
      } else {
         return mid;
      }
   }
   return -1; // not found
}

int vftr_linear_search_callee(vftr_stack_t *stacks, int callerID, uintptr_t address) {
   vftr_stack_t stack = stacks[callerID];
   int calleeID = -1;
   for (int icallee=0; icallee<stack.ncallees; icallee++) {
      int stackID = stack.callees[icallee];
      if (address == stacks[stackID].address) {
         calleeID = stackID;
         break;
      }
   }

   return calleeID;
}
