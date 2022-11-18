#ifndef CONFIGURATION_TYPES_H
#define CONFIGURATION_TYPES_H

#include <stdbool.h>
#include <regex.h>

// Reusable definitions
#define CONFIG_STRUCT_DEFAULTS \
   char *name; \
   bool set;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   bool value;
} config_bool_t;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   int value;
} config_int_t;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   double value;
} config_float_t;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   char *value;
} config_string_t;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   char *value;
   regex_t *regex;
} config_regex_t;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   config_string_t column;
   config_bool_t ascending;
} config_sort_table_t;

// Section definitions
typedef struct {
   CONFIG_STRUCT_DEFAULTS
   config_bool_t show_table;
   config_bool_t show_calltime_imbalances;
   config_bool_t show_callpath;
   config_bool_t show_overhead;
   config_sort_table_t sort_table;
} config_profile_table_t;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   config_bool_t show_table;
   config_int_t max_stack_ids;
   config_sort_table_t sort_table;
} config_name_grouped_profile_table_t;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   config_bool_t active;
   config_float_t sample_interval;
   config_int_t outbuffer_size;
   config_regex_t precise_functions;
} config_sampling_t;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   config_bool_t show_table;
   config_bool_t log_messages;
   config_string_t only_for_ranks;
   config_bool_t show_sync_time;
   config_bool_t show_callpath;
   config_sort_table_t sort_table;
} config_mpi_t;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   config_bool_t show_table;
   config_sort_table_t sort_table;
} config_cuda_t;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   config_bool_t show_table;
   config_sort_table_t sort_table;
} config_accprof_t;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   config_bool_t active;
} config_hardware_scenarios_t;

typedef struct {
   config_bool_t off;
   config_string_t output_directory;
   config_string_t outfile_basename;
   config_string_t logfile_for_ranks;
   config_bool_t print_config;
   config_bool_t strip_module_names;
   config_bool_t demangle_cxx;
   config_profile_table_t profile_table;
   config_name_grouped_profile_table_t name_grouped_profile_table;
   config_sampling_t sampling;
   config_mpi_t mpi;
   config_cuda_t cuda;
   config_accprof_t accprof;
   config_hardware_scenarios_t hardware_scenarios;
   bool valid;
   char *config_file_path;
} config_t;

#undef CONFIG_STRUCT_DEFAULTS
#endif
