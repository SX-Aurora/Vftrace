#include <stdio.h>

#include "bool_strings.h"
#include "configuration_types.h"
#include "configuration_defaults.h"

void vftr_print_config_indent(FILE *fp, int level) {
   fprintf(fp, "%*s", 3*level, "");
}

void vftr_print_config_bool(FILE *fp, int level, config_bool_t cfg_bool) {
   level++;
   vftr_print_config_indent(fp, level);
   fprintf(fp, "\"%s\": %s", cfg_bool.name, vftr_bool_to_string(cfg_bool.value));
}

void vftr_print_config_int(FILE *fp, int level, config_int_t cfg_int) {
   level++;
   vftr_print_config_indent(fp, level);
   fprintf(fp, "\"%s\": %d", cfg_int.name, cfg_int.value);
}

void vftr_print_config_float(FILE *fp, int level, config_float_t cfg_float) {
   level++;
   vftr_print_config_indent(fp, level);
   fprintf(fp, "\"%s\": %lf", cfg_float.name, cfg_float.value);
}

void vftr_print_config_string(FILE *fp, int level, config_string_t cfg_string) {
   level++;
   vftr_print_config_indent(fp, level);
   fprintf(fp, "\"%s\": ", cfg_string.name);
   if (cfg_string.value == NULL) {
      fprintf(fp, "null");
   } else {
      fprintf(fp, "\"%s\"", cfg_string.value);
   }
}

void vftr_print_config_regex(FILE *fp, int level, config_regex_t cfg_regex) {
   level++;
   vftr_print_config_indent(fp, level);
   fprintf(fp, "\"%s\": ", cfg_regex.name);
   if (cfg_regex.value == NULL) {
      fprintf(fp, "null");
   } else {
      fprintf(fp, "\"%s\"", cfg_regex.value);
   }
}

void vftr_print_config_sort_table(FILE *fp, int level,
                                  config_sort_table_t cfg_sort_table) {
   level++;
   vftr_print_config_indent(fp, level);
   fprintf(fp, "\"%s\": {\n", cfg_sort_table.name);
   vftr_print_config_string(fp, level, cfg_sort_table.column);
   fprintf(fp,",\n");
   vftr_print_config_bool(fp, level, cfg_sort_table.ascending);
   fprintf(fp,"\n");
   vftr_print_config_indent(fp, level);
   fprintf(fp,"}");
}

void vftr_print_config_profile_table(FILE *fp, int level,
                                     config_profile_table_t cfg_profile_table) {
   level++;
   vftr_print_config_indent(fp, level);
   fprintf(fp, "\"%s\": {\n", cfg_profile_table.name);
   vftr_print_config_bool(fp, level, cfg_profile_table.show_table);
   fprintf(fp,",\n");
   vftr_print_config_bool(fp, level, cfg_profile_table.show_calltime_imbalances);
   fprintf(fp,",\n");
   vftr_print_config_bool(fp, level, cfg_profile_table.show_callpath);
   fprintf(fp,",\n");
   vftr_print_config_bool(fp, level, cfg_profile_table.show_overhead);
   fprintf(fp,",\n");
   vftr_print_config_sort_table(fp, level, cfg_profile_table.sort_table);
   fprintf(fp,"\n");
   vftr_print_config_indent(fp, level);
   fprintf(fp,"}");
}

void vftr_print_config_name_grouped_profile_table(FILE *fp, int level,
                                                  config_name_grouped_profile_table_t
                                                  cfg_profile_table) {
   level++;
   vftr_print_config_indent(fp, level);
   fprintf(fp, "\"%s\": {\n", cfg_profile_table.name);
   vftr_print_config_bool(fp, level, cfg_profile_table.show_table);
   fprintf(fp,",\n");
   vftr_print_config_int(fp, level, cfg_profile_table.max_stack_ids);
   fprintf(fp,",\n");
   vftr_print_config_sort_table(fp, level, cfg_profile_table.sort_table);
   fprintf(fp,"\n");
   vftr_print_config_indent(fp, level);
   fprintf(fp,"}");
}

void vftr_print_config_sampling(FILE *fp, int level, config_sampling_t cfg_sampling) {
   level++;
   vftr_print_config_indent(fp, level);
   fprintf(fp, "\"%s\": {\n", cfg_sampling.name);
   vftr_print_config_bool(fp, level, cfg_sampling.active);
   fprintf(fp,",\n");
   vftr_print_config_float(fp, level, cfg_sampling.sample_interval);
   fprintf(fp,",\n");
   vftr_print_config_int(fp, level, cfg_sampling.outbuffer_size);
   fprintf(fp,",\n");
   vftr_print_config_regex(fp, level, cfg_sampling.precise_functions);
   fprintf(fp,"\n");
   vftr_print_config_indent(fp, level);
   fprintf(fp,"}");
}

void vftr_print_config_mpi(FILE *fp, int level, config_mpi_t cfg_mpi) {
   level++;
   vftr_print_config_indent(fp, level);
   fprintf(fp, "\"%s\": {\n", cfg_mpi.name);
   vftr_print_config_bool(fp, level, cfg_mpi.show_table);
   fprintf(fp,",\n");
   vftr_print_config_bool(fp, level, cfg_mpi.log_messages);
   fprintf(fp,",\n");
   vftr_print_config_string(fp, level, cfg_mpi.only_for_ranks);
   fprintf(fp,",\n");
   vftr_print_config_bool(fp, level, cfg_mpi.show_sync_time);
   fprintf(fp,",\n");
   vftr_print_config_bool(fp, level, cfg_mpi.show_callpath);
   fprintf(fp,",\n");
   vftr_print_config_sort_table(fp, level, cfg_mpi.sort_table);
   fprintf(fp,"\n");
   vftr_print_config_indent(fp, level);
   fprintf(fp,"}");
}

void vftr_print_config_cuda(FILE *fp, int level, config_cuda_t cfg_cuda) {
   level++;
   vftr_print_config_indent(fp, level);
   fprintf(fp, "\"%s\": {\n", cfg_cuda.name);
   vftr_print_config_bool(fp, level, cfg_cuda.show_table);
   fprintf(fp,",\n");
   vftr_print_config_sort_table(fp, level, cfg_cuda.sort_table);
   fprintf(fp,"\n");
   vftr_print_config_indent(fp, level);
   fprintf(fp,"}");
}

void vftr_print_config_hardware_scenarios(FILE *fp, int level,
                                          config_hardware_scenarios_t
                                          cfg_hardware_scenarios) {
   level++;
   vftr_print_config_indent(fp, level);
   fprintf(fp, "\"%s\": {\n", cfg_hardware_scenarios.name);
   vftr_print_config_bool(fp, level, cfg_hardware_scenarios.active);
   fprintf(fp,"\n");
   vftr_print_config_indent(fp, level);
   fprintf(fp,"}");
}

void vftr_print_config(FILE *fp, config_t config, bool show_title) {
   if (show_title) {
      fprintf(fp, "\n");
      if (config.config_file_path == NULL) {
         fprintf(fp, "Vftrace default configuration:\n");
      } else {
         fprintf(fp, "Vftrace configuration read from \"%s\"\n",
                 config.config_file_path);
      }
   }
   int level = 0;
   fprintf(fp, "{\n");
   vftr_print_config_bool(fp, level, config.off);
   fprintf(fp, ",\n");
   vftr_print_config_string(fp, level, config.output_directory);
   fprintf(fp, ",\n");
   vftr_print_config_string(fp, level, config.outfile_basename);
   fprintf(fp, ",\n");
   vftr_print_config_string(fp, level, config.logfile_for_ranks);
   fprintf(fp, ",\n");
   vftr_print_config_bool(fp, level, config.print_config);
   fprintf(fp, ",\n");
   vftr_print_config_bool(fp, level, config.strip_module_names);
   fprintf(fp, ",\n");
   vftr_print_config_bool(fp, level, config.demangle_cxx);
   fprintf(fp, ",\n");
   vftr_print_config_profile_table(fp, level, config.profile_table);
   fprintf(fp, ",\n");
   vftr_print_config_name_grouped_profile_table(fp, level,
                                                config.name_grouped_profile_table);
   fprintf(fp, ",\n");
   vftr_print_config_sampling(fp, level, config.sampling);
   fprintf(fp, ",\n");
   vftr_print_config_mpi(fp, level, config.mpi);
   fprintf(fp, ",\n");
   vftr_print_config_cuda(fp, level, config.cuda);
   fprintf(fp, ",\n");
   vftr_print_config_hardware_scenarios(fp, level, config.hardware_scenarios);
   fprintf(fp, "\n");
   fprintf(fp, "}\n");
}
