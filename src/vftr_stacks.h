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
   bool print_profile;
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


// Synchronise stack-IDs between processes
void vftr_normalize_stacks();

// Write the stacks out
void vftr_write_stacks_vfd (FILE *fp, int level, function_t *func);
void vftr_write_stack_ascii (FILE *fp, double time, function_t *func, char *label, int timeToSample);
void vftr_print_local_stacklist (function_t **funcTable, FILE *pout, int ntop);
void vftr_print_local_demangled (function_t **funcTable, FILE *pout, int ntop);
void vftr_print_global_stacklist (FILE *pout);

typedef struct stack_leaf {
	int stack_id;
	int func_id;
	int final_id;
	struct stack_leaf *next_in_level;
	struct stack_leaf *callee;	
	struct stack_leaf *origin;
} stack_leaf_t;	

// The headers of the stacktree table.
// The enums below allow for an access to fields without having to care about the actual number, e.g.
// when the order of elements is rearranged.
extern const char *vftr_stacktree_headers[6];
enum column_ids {TIME, CALLS, IMBA, SEND_BYTES, RECV_BYTES, STACK_ID};

int vftr_stack_length (int stack_id0);
void vftr_fill_into_stack_tree (stack_leaf_t **this_leaf, int n_stack_ids, int *stacks_ids, int func_id);
void vftr_stack_compute_imbalances (double *imbalances, int n_final_stack_ids, int *final_stack_ids);
void vftr_stack_get_total_time (stack_leaf_t *leaf, long long *total_time);
void vftr_create_stacktree (stack_leaf_t **stack_tree, int n_final_stack_ids, int *final_stack_ids);
void vftr_scan_stacktree (stack_leaf_t *stack_tree, int n_final_stack_ids, double *imbalances,
			  double *t_max, int *n_calls_max, double *imba_max, int *n_spaces_max, int *n_chars_max);
void vftr_print_function_stack (FILE *fp, char *func_name, int n_final_stack_ids,
			        double *imbalances, long long total_time,
			        double t_max, int n_calls_max, double imba_max, int n_spaces_max, 
			        stack_leaf_t *stack_tree);

typedef struct stack_string {
   char *s;
   int len; 
   int id;
   int depth;
} stack_string_t;

extern stack_string_t *vftr_global_stack_strings;
void vftr_create_global_stack_strings ();
void vftr_create_stack_string (int i_stack, char **name, int *len, int *depth);

// test functions
int vftr_stacks_test_1(FILE *fp_in, FILE *fp_out);
int vftr_stacks_test_2(FILE *fp_in, FILE *fp_out);

#endif
