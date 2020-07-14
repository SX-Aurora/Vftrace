#ifndef VFTR_FUNCTIONS_H
#define VFTR_FUNCTIONS_H

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct ProfFragment {
   long long cycles;
   struct Function *func;
   struct ProfFragment *next;
} pfrag_t;

typedef struct ProfileData {
   // amount of calls 
   long long calls;
   // cycles spend in the function (excluding subfunctions)
   long long cycles;
   // cycles spend in the function (including subfunctions)
   long long cycInc;
   // time spend in the function (excluding subfunctions)
   long long timeExcl;
   // time speind in the function (including subfunctions)
   long long timeIncl;
   // 
   long long flops, *event_count, *events[2], ecreads;
   //
   pfrag_t *first,*last;
   //
   int pfcount, ic;
} profdata_t;

typedef struct Function {
   // pointers to other functions in the stack
   struct Function *first, *next, *call, *ret, *root;
   // the address of the function
   void *address;
   // name of the function
   char *name;
   // string with the full callstack 
   char *full;
   // profiling data
   profdata_t      *prof_current, *prof_previous;
   pfrag_t         *frag;
   // is this function measured precicely?
   bool precise;
   // local and global stack-ID
   int id, gid;
   bool profile_this, exclude_this;
   int new, openmp, detail, levels,
       recursion_depth, line_beg, line_end;
   // Unique hash of the callstack 
   // needed vor stack comparison among processes
   uint64_t stackHash;
} function_t;

// add a new function to the stack tables
function_t *vftr_new_function(void *arg, const char *function_name,
                              function_t *caller, char *info, int line,
                              bool isPrecise);

// Reset all function internal counters
void vftr_reset_counts (int me, function_t *func);
#endif
