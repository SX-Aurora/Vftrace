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

#ifndef VFTR_STACKS_H
#define VFTR_STACKS_H

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include <stdbool.h>
#include "vftr_timer.h"
#include "vftr_functions.h"

// number of omp threads
extern int vftr_omp_threads;

// Maximum time in a call tree, searched for in vftr_finalize
extern long long vftr_maxtime;

// Stack information on local and global scale
// TODO: fuse the stack info types
typedef struct StackInfo {
   // id of the calling function
   int  ret;
   // function name string of the current function
   char name[80];
} stackinfo_t;

typedef struct GStackInfo {
   // global id of the calling function
   int  ret;
   // local id of the current function
   int locID;
   // function name string of the current function
   char *name;
} gstackinfo_t;

// Profiling structs

struct Performance {
    unsigned long long hits;   /* Some internal performance data */
    unsigned long long misses;
    unsigned long long steps;
};

// number of locally unique stacks
extern int vftr_stackscount;
// number of globally unique stacks
extern int vftr_gStackscount;

// Collective information about all stacks across processes
extern gstackinfo_t  *vftr_gStackinfo;

// Table of all functions defined
extern function_t **vftr_func_table;
// Size of vftr_func_table (grows as needed)
extern unsigned int  vftr_func_table_size;

// Function call stack
//extern function_t **vftr_fstack;
extern function_t *vftr_fstack;
// Function call stack roots
extern function_t *vftr_froots;
// Profile data
extern struct Performance *vftr_prof;
// Profile data sample
extern profdata_t vftr_prof_data;

// initialize stacks only called from vftr_initialize
void vftr_initialize_stacks();

// Write the stacks out
void vftr_write_stacks (FILE *fp, int level, function_t *func);

// Synchronise stack-IDs between processes
int vftr_normalize_stacks();

void vftr_print_stack (int tid, double time, function_t *func, char *label, int timeToSample);
void vftr_print_local_stacklist (function_t **funcTable, FILE *pout, int ntop);
void vftr_print_local_demangled (function_t **funcTable, FILE *pout, int ntop);
void vftr_print_global_stacklist (FILE *pout);

#endif
