#include <stdlib.h>
#include <stdbool.h>

#include <string.h>

#include "configuration_types.h"
#include "regular_expressions.h"
#include "range_expand.h"

config_bool_t vftr_set_config_bool_default(char *name, bool value) {
   config_bool_t cfg_bool;
   cfg_bool.name = strdup(name);
   cfg_bool.set = false;
   cfg_bool.value = value;
   return cfg_bool;
}

config_int_t vftr_set_config_int_default(char *name, int value) {
   config_int_t cfg_int;
   cfg_int.name = strdup(name);
   cfg_int.set = false;
   cfg_int.value = value;
   return cfg_int;
}

config_float_t vftr_set_config_float_default(char *name, double value) {
   config_float_t cfg_float;
   cfg_float.name = strdup(name);
   cfg_float.set = false;
   cfg_float.value = value;
   return cfg_float;
}

config_string_t vftr_set_config_string_default(char *name, char *value) {
   config_string_t cfg_string;
   cfg_string.name = strdup(name);
   cfg_string.set = false;
   cfg_string.value = value == NULL ? NULL : strdup(value);
   return cfg_string;
}

config_regex_t vftr_set_config_regex_default(char *name) {
   config_regex_t cfg_regex;
   cfg_regex.name = strdup(name);
   cfg_regex.set = false;
   cfg_regex.value = NULL;
   cfg_regex.regex = NULL;
   return cfg_regex;
}

config_sort_table_t vftr_set_config_sort_table_default(char *column, bool ascending) {
   config_sort_table_t cfg_sort_table;
   cfg_sort_table.name = strdup("sort_table");
   cfg_sort_table.set = false;
   cfg_sort_table.column =
      vftr_set_config_string_default("column", column);
   cfg_sort_table.ascending =
      vftr_set_config_bool_default("ascending", ascending);
   return cfg_sort_table;
}

config_profile_table_t vft_set_config_profile_table_default() {
   config_profile_table_t cfg_profile_table;
   cfg_profile_table.name = strdup("profile_table");
   cfg_profile_table.set = false;
   cfg_profile_table.active =
      vftr_set_config_bool_default("active", true);
   cfg_profile_table.show_calltime_imbalances =
      vftr_set_config_bool_default("show_calltime_imbalances", false);
   cfg_profile_table.show_callpath =
      vftr_set_config_bool_default("show_callpath", false);
   cfg_profile_table.show_overhead =
      vftr_set_config_bool_default("show_overhead", false);
   cfg_profile_table.sort_table =
      vftr_set_config_sort_table_default("time_excl", false);
   return cfg_profile_table;
}

config_name_grouped_profile_table_t vftr_set_config_name_grouped_profile_table_default() {
   config_name_grouped_profile_table_t cfg_profile_table;
   cfg_profile_table.name = strdup("name_grouped_profile_table");
   cfg_profile_table.set = false;
   cfg_profile_table.active =
      vftr_set_config_bool_default("active", false);
   cfg_profile_table.max_stack_ids =
      vftr_set_config_int_default("max_stack_ids", 8);
   cfg_profile_table.sort_table =
      vftr_set_config_sort_table_default("time_excl", false);
   return cfg_profile_table;
}

config_sampling_t vftr_set_config_sampling_default() {
   config_sampling_t cfg_sampling;
   cfg_sampling.name = strdup("sampling");
   cfg_sampling.set = false;
   cfg_sampling.active =
      vftr_set_config_bool_default("active", false);
   cfg_sampling.sample_interval =
      vftr_set_config_float_default("sample_interval", 0.005);
   cfg_sampling.outbuffer_size =
      vftr_set_config_int_default("outbuffer_size", 8);
   cfg_sampling.precise_functions =
      vftr_set_config_regex_default("precise_functions");
   return cfg_sampling;
}

config_mpi_t vftr_set_config_mpi_default() {
   config_mpi_t cfg_mpi;
   cfg_mpi.name = strdup("mpi");
   cfg_mpi.set = false;
   cfg_mpi.active =
      vftr_set_config_bool_default("active", false);
   cfg_mpi.only_for_ranks =
      vftr_set_config_string_default("only_for_ranks", "all");
   cfg_mpi.show_sync_time =
      vftr_set_config_bool_default("show_sync_time", false);
   cfg_mpi.show_callpath =
      vftr_set_config_bool_default("show_callpath", false);
   cfg_mpi.sort_table =
      vftr_set_config_sort_table_default("none", false);
   return cfg_mpi;
}

config_cuda_t vftr_set_config_cuda_default() {
   config_cuda_t cfg_cuda;
   cfg_cuda.name = strdup("cuda");
   cfg_cuda.set = false;
   cfg_cuda.active =
      vftr_set_config_bool_default("active", false);
   cfg_cuda.sort_table =
      vftr_set_config_sort_table_default("time", false);
   return cfg_cuda;
}

config_hardware_scenarios_t vftr_set_config_hardware_scenarios_default() {
   config_hardware_scenarios_t cfg_hardware_scenarios;
   cfg_hardware_scenarios.name = strdup("hardware_scenarios");
   cfg_hardware_scenarios.set = false;
   cfg_hardware_scenarios.active =
      vftr_set_config_bool_default("active", false);
   return cfg_hardware_scenarios;
}

config_t vftr_set_config_default() {
   config_t config;
   config.off =
      vftr_set_config_bool_default("off", false);
   config.output_directory =
      vftr_set_config_string_default("output_directory", ".");
   config.outfile_basename =
      vftr_set_config_string_default("outfile_basename", NULL);
   config.logfile_for_ranks =
      vftr_set_config_string_default("logfile_for_ranks", "none");
   config.print_config =
      vftr_set_config_bool_default("print_config", true);
   config.strip_module_names =
      vftr_set_config_bool_default("strip_module_names", false);
   config.demangle_cxx =
      vftr_set_config_bool_default("demangle_cxx", false);
   config.profile_table =
      vft_set_config_profile_table_default();
   config.name_grouped_profile_table =
      vftr_set_config_name_grouped_profile_table_default();
   config.sampling =
      vftr_set_config_sampling_default();
   config.mpi =
      vftr_set_config_mpi_default();
   config.cuda =
      vftr_set_config_cuda_default();
   config.hardware_scenarios =
      vftr_set_config_hardware_scenarios_default();
   config.config_file_path = NULL;
   config.valid = true;
   return config;
}

