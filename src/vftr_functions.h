/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#ifndef VFTR_FUNCTIONS_H
#define VFTR_FUNCTIONS_H

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct ProfileData {
   // amount of calls 
   long long calls;
   // cycles spend in the function (excluding subfunctions)
   long long cycles;
   // time spend in the function (excluding subfunctions)
   long long timeExcl;
   // time spend in the function (including subfunctions)
   long long timeIncl;
   // 
   long long flops, *event_count, *events[2], ecreads;
   //
   int pfcount, ic;
} profdata_t;

typedef struct Function {
   // pointers to other functions in the stack
   struct Function *first_in_level, *next_in_level, *callee, *return_to, *root;
   // the address of the function
   void *address;
   // name of the function
   char *name;
   // string with the full callstack 
   char *full;
   // profiling data
   profdata_t prof_current, prof_previous;
   // is this function measured precisely?
   bool precise;
   // local and global stack-ID
   int id, gid;
   bool profile_this, exclude_this;
   bool new, detail;
   int levels, recursion_depth, line_beg, line_end;
   // Unique hash of the callstack 
   // needed vor stack comparison among processes
   uint64_t stackHash;
} function_t;

void vftr_find_function_in_table (char *func_name, int **indices, int *n_indices, bool to_lower_case);
void vftr_find_function_in_stack (char *func_name, int **indices, int *n_indices, bool to_lower_case);

// Remove everything in front of (and including) _MP_ for all the symbols in
// the table, if necessary.
void vftr_strip_all_module_names ();

// add a new function to the stack tables
function_t *vftr_new_function(void *arg, const char *function_name,
                              function_t *caller, int line, bool is_precise);

// Reset all function internal counters
void vftr_reset_counts (function_t *func);

// test functions
int vftr_functions_test_1 (FILE *fp_in, FILE *fp_out);
int vftr_functions_test_2 (FILE *fp_in, FILE *fp_out);
int vftr_functions_test_3 (FILE *fp_in, FILE *fp_out);
int vftr_functions_test_4 (FILE *fp_in, FILE *fp_out);
int vftr_functions_test_5 (FILE *fp_in, FILE *fp_out);
#endif
