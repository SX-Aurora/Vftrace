#ifndef ENVIRONMENT_TYPES_H
#define ENVIRONMENT_TYPES_H

#include <stdbool.h>

#include <regex.h>

typedef enum {
   env_none,
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
   // in order to access the environment variables
   // like an array by index, all non-environment variables
   // need to be at the end of the struct!
   env_var_t vftrace_off;
   env_var_t do_sampling;
   env_var_t output_directory;
   env_var_t logfile_basename;
   env_var_t logfile_for_ranks;
   env_var_t ranks_in_mpi_profile;
   env_var_t sampletime;
   env_var_t mpi_log;
   env_var_t mpi_show_sync_time;
   env_var_t vfd_bufsize;
   env_var_t include_only_regex;
   env_var_t scenario_file;
   env_var_t preciseregex;
   env_var_t print_stack_profile;
   env_var_t print_stacks_for;
   env_var_t strip_module_names;
   env_var_t sort_profile_table;
   env_var_t sort_mpi_table;
   env_var_t show_overhead;
   env_var_t show_calltime_imbalances;
   env_var_t print_environment;
   env_var_t callpath_in_profile;
   env_var_t callpath_in_mpi_profile;
   env_var_t demangle_cxx;
   bool valid;
   int nenv_vars;
} environment_t;

#endif
