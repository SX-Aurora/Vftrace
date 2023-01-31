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
   vftr_print_config_bool(fp, level, cfg_profile_table.show_minmax_summary);
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

void vftr_print_config_accprof (FILE *fp, int level, config_accprof_t cfg_accprof) {
   level++;
   vftr_print_config_indent(fp, level);
   fprintf(fp, "\"%s\": {\n", cfg_accprof.name);
   vftr_print_config_bool(fp, level, cfg_accprof.show_table);
   fprintf(fp,",\n");
   vftr_print_config_bool(fp, level, cfg_accprof.show_event_details);
   fprintf(fp,",\n");
   vftr_print_config_sort_table(fp, level, cfg_accprof.sort_table);
   fprintf(fp,"\n");
   vftr_print_config_indent(fp, level);
   fprintf(fp,"}");
}

void vftr_print_config_hwcounters (FILE *fp, int level, config_hwcounters_t cfg_hwc) {
   int n_max = cfg_hwc.hwc_name.n_elements;
   n_max = cfg_hwc.symbol.n_elements > n_max ? cfg_hwc.symbol.n_elements : n_max;
   int i_name = 0;
   int i_symbol = 0;
   level++;
   vftr_print_config_indent(fp, level);
   fprintf (fp, "\"%s\": ", cfg_hwc.name);
   fprintf (fp, "[\n");
   level++;
   for (int i = 0; i < n_max; i++) {
      vftr_print_config_indent(fp, level);
      fprintf (fp, "{\n");
      if (cfg_hwc.hwc_name.list_idx[i_name] == i) {
         vftr_print_config_indent(fp, level);
         fprintf (fp, "\"%s\": \"%s\",\n", cfg_hwc.hwc_name.name,
                                           cfg_hwc.hwc_name.values[i_name]);
         i_name++;
      }
      if (cfg_hwc.symbol.list_idx[i_symbol] == i) {
         vftr_print_config_indent(fp, level);
         fprintf (fp, "\"%s\": \"%s\",\n", cfg_hwc.symbol.name,
                                           cfg_hwc.symbol.values[i_symbol]);
         i_symbol++;
      }
      vftr_print_config_indent(fp, level);
      fprintf (fp, "},\n");
   }
   level--;
   vftr_print_config_indent(fp, level);
   fprintf (fp, "]");
}

void vftr_print_config_hwobservables (FILE *fp, int level, config_hwobservables_t cfg_hwobs) {
   int n_max = cfg_hwobs.obs_name.n_elements > cfg_hwobs.formula_expr.n_elements ? 
               cfg_hwobs.obs_name.n_elements : cfg_hwobs.formula_expr.n_elements;
   n_max = cfg_hwobs.unit.n_elements > n_max ? cfg_hwobs.unit.n_elements : n_max;
   int i_obs = 0;
   int i_formula = 0;
   int i_unit = 0;
   level++;
   vftr_print_config_indent(fp, level);
   fprintf (fp, "\"%s\": ", cfg_hwobs.name);
   fprintf (fp, "[\n");
   level++;
   for (int i = 0; i < n_max; i++) {
      vftr_print_config_indent(fp, level);
      fprintf (fp, "{\n");
      if (cfg_hwobs.obs_name.list_idx[i_obs] == i) {
         vftr_print_config_indent(fp, level);
         fprintf (fp, "\"%s\": \"%s\",\n", cfg_hwobs.obs_name.name,
                                           cfg_hwobs.obs_name.values[i_obs]);
         i_obs++;
      }
      if (cfg_hwobs.formula_expr.list_idx[i_formula] == i) {
         vftr_print_config_indent(fp, level);
         fprintf (fp, "\"%s\": \"%s\",\n", cfg_hwobs.formula_expr.name,
                                           cfg_hwobs.formula_expr.values[i_formula]);
         i_formula++;
      }
      if (cfg_hwobs.unit.list_idx[i_unit] == i) {
         vftr_print_config_indent(fp, level);
         fprintf (fp, "\"%s\": \"%s\",\n", cfg_hwobs.unit.name,
                                           cfg_hwobs.unit.values[i_unit]);
         i_unit++;
      }
      vftr_print_config_indent(fp, level);
      fprintf (fp, "},\n");
   }
   level--;
   vftr_print_config_indent(fp, level);
   fprintf (fp, "]");
}

void vftr_print_config_hwprof (FILE *fp, int level, config_hwprof_t cfg_hwprof) {
   level++;
   vftr_print_config_indent(fp, level);
   fprintf (fp, "\"%s\": {\n", cfg_hwprof.name);
   vftr_print_config_string (fp, level, cfg_hwprof.hwc_type);
   fprintf (fp, ",\n");
   vftr_print_config_bool (fp, level, cfg_hwprof.active);
   fprintf (fp, ",\n");
   vftr_print_config_bool (fp, level, cfg_hwprof.show_observables);
   fprintf (fp, ",\n");
   vftr_print_config_bool (fp, level, cfg_hwprof.show_counters);
   fprintf (fp, ",\n");
   vftr_print_config_bool (fp, level, cfg_hwprof.show_summary);
   fprintf (fp, ",\n");
   vftr_print_config_int (fp, level, cfg_hwprof.sort_by_column);
   fprintf (fp, ",\n");
   vftr_print_config_string (fp, level, cfg_hwprof.default_scenario);
   fprintf (fp, ",\n");
   vftr_print_config_hwcounters (fp, level, cfg_hwprof.counters);
   fprintf (fp, ",\n");
   vftr_print_config_hwobservables (fp, level, cfg_hwprof.observables);
   fprintf (fp, "\n");
   vftr_print_config_indent(fp, level);
   fprintf (fp, "}"); 
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
   vftr_print_config_bool(fp, level, config.include_cxx_prelude);
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
   vftr_print_config_hwprof(fp, level, config.hwprof);
   fprintf(fp, "\n");
   fprintf(fp, "}\n");
}
