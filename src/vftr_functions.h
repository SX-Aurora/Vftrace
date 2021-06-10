#ifndef VFTR_FUNCTIONS_H
#define VFTR_FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct memProf {
   long long mem_entry;
   long long mem_exit;
   long long mem_max;
   long long next_memtrace_entry;
   long long next_memtrace_exit;
   int mem_tolerance;
   int mem_increment;
} mem_prof_t;

typedef struct ProfileData {
   // amount of calls 
   long long calls;
   // cycles spend in the function (excluding subfunctions)
   long long cycles;
   // time spend in the function (excluding subfunctions)
   long long time_excl;
   // time spend in the function (including subfunctions)
   long long time_incl;
   // 
   long long *event_count, *events[2];
   //
   int ic;
   long mpi_tot_send_bytes;
   long mpi_tot_recv_bytes;
   mem_prof_t *mem_prof;
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
   bool profile_this;
   bool new, detail;
   int levels, recursion_depth;
   // Unique hash of the callstack 
   // needed vor stack comparison among processes
   uint64_t stackHash;
   long long overhead;
   bool open;
} function_t;

void vftr_find_function_in_table (char *func_name, int **indices, int *n_indices, bool to_lower_case);
void vftr_find_function_in_stack (char *func_name, int **indices, int *n_indices, bool to_lower_case);

// Remove everything in front of (and including) _MP_ for all the symbols in
// the table, if necessary.
void vftr_strip_all_module_names ();

void vftr_demangle_all_func_names ();

// add a new function to the stack tables
function_t *vftr_new_function(void *arg, const char *function_name, function_t *caller, bool is_precise);

// Reset all function internal counters
void vftr_reset_counts (function_t *func);

void vftr_write_function (FILE *fp, function_t *func, bool verbose);

struct loc_glob_id {
   int loc;
   int glob;
};

//extern int *vftr_print_stackid_list;
extern struct loc_glob_id *vftr_print_stackid_list;
extern int vftr_n_print_stackids;
extern int vftr_stackid_list_size; 
#define STACKID_LIST_INC 50

void vftr_stackid_list_init ();
void vftr_stackid_list_add (int local_stack_id, int global_stack_id);
void vftr_stackid_list_print (FILE *fp);
void vftr_stackid_list_finalize ();

void vftr_sample_vmrss (long long n_calls, bool is_entry, bool verbose, mem_prof_t *mem_prof);
double vftr_get_max_memory (function_t *func);

// test functions
int vftr_functions_test_3 (FILE *fp_in, FILE *fp_out);
int vftr_functions_test_4 (FILE *fp_in, FILE *fp_out);
int vftr_functions_test_5 (FILE *fp_in, FILE *fp_out);
#endif
