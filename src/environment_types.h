#ifndef ENVIRONMENT_TYPES_H
#define ENVIRONMENT_TYPES_H

#include <stdbool.h>
#include <regex.h>

typedef enum {
   env_int,
   env_long,
   env_longlong,
   env_float,
   env_double,
   env_bool,
   env_string,
   env_regex
} env_var_kind;

typedef union {
   int int_val;
   long long_val;
   long long longlong_val;
   float float_val;
   double double_val;
   bool bool_val;
   char *string_val;
   regex_t *regex_val;
} env_var_value;

typedef struct {
   env_var_kind value_kind;
   env_var_value value;
   char *value_string;
   char *name;
   bool set;
} env_var_t;

typedef struct {
   bool valid;
   env_var_t vftrace_off;
   env_var_t do_sampling;
   env_var_t regions_precise;
   env_var_t output_directory;
   env_var_t logfile_basename;
   env_var_t logfile_for_ranks;
   env_var_t mpi_summary_for_ranks;
   env_var_t sampletime;
   env_var_t stoptime;
   env_var_t accurate_profile;
   env_var_t prof_truncate;
   env_var_t prof_truncate_cutoff;
   env_var_t mpi_log;
   env_var_t mpi_show_sync_time;
   env_var_t signals_off;
   env_var_t bufsize;
   env_var_t runtime_profile_funcs;
   env_var_t include_only_regex;
   env_var_t detail_until_cum_cycles;
   env_var_t scenario_file;
   env_var_t preciseregex;
   env_var_t print_stack_profile;
   env_var_t license_verbose;
   env_var_t print_stacks_for;
   env_var_t print_loadinfo_for;
   env_var_t strip_module_names;
   env_var_t create_html;
   env_var_t sort_profile_table;
   env_var_t show_overhead;
   env_var_t meminfo_method;
   env_var_t meminfo_stepsize;
   env_var_t print_env;
   env_var_t no_memtrace;
   env_var_t show_stacks_in_profile;
   env_var_t no_stack_normalization;
   env_var_t demangle_cpp;
   env_var_t show_startup;
} environment_t;

#endif 
