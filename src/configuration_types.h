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
   int n_elements;
   char **values;
   int *list_idx;
} config_string_list_t;

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
   config_bool_t show_minmax_summary;
   config_sort_table_t sort_table;
   config_bool_t separate;
} config_profile_table_t;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   config_bool_t show_stacklist;
} config_stacklist_t;

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
   config_bool_t active;
   config_bool_t show_table;
   config_sort_table_t sort_table;
   config_bool_t show_event_details;
} config_accprof_t;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   config_string_list_t hwc_name;
   config_string_list_t symbol;
} config_hwcounters_t;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   config_string_list_t obs_name;
   config_string_list_t formula_expr;
   config_string_list_t unit;
} config_hwobservables_t;

typedef struct {
   CONFIG_STRUCT_DEFAULTS
   config_bool_t active;
   config_string_t hwc_type;
   config_bool_t show_observables;
   config_bool_t show_counters;
   config_bool_t show_summary;
   config_int_t sort_by_column;
   config_string_t default_scenario;
   config_hwcounters_t counters;
   config_hwobservables_t observables;
} config_hwprof_t;

typedef struct {
   config_bool_t off;
   config_string_t output_directory;
   config_string_t outfile_basename;
   config_string_t logfile_for_ranks;
   config_bool_t print_config;
   config_bool_t strip_module_names;
   config_bool_t demangle_cxx;
   config_bool_t include_cxx_prelude;
   config_profile_table_t profile_table;
   config_stacklist_t stacklist;
   config_name_grouped_profile_table_t name_grouped_profile_table;
   config_sampling_t sampling;
   config_mpi_t mpi;
   config_cuda_t cuda;
   config_accprof_t accprof;
   config_hwprof_t hwprof;
   bool valid;
   char *config_file_path;
} config_t;

#undef CONFIG_STRUCT_DEFAULTS
#endif
