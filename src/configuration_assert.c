#include <stdlib.h>
#include <stdio.h>

#include <unistd.h>
#include <sys/stat.h>
#include <string.h>

#include "signal_handling.h"
#include "configuration_types.h"
#include "range_expand.h"

#include "hwprof_state_types.h"

// General assertion routines
void vftr_config_active_assert(FILE *fp, config_bool_t cfg_active) {
   (void) fp;
   (void) cfg_active;
}

void vftr_config_show_table_assert(FILE *fp, config_bool_t cfg_show_table) {
   (void) fp;
   (void) cfg_show_table;
}

void vftr_config_column_assert(FILE *fp, config_string_t cfg_column,
                               int nvalid_columns,
                               const char *valid_columns[]) {
   if (cfg_column.value == NULL) {
      fprintf(fp, "sort_table.column cannot be \"null\"\n");
      vftr_abort(0);
   }
   bool colum_is_valid = false;
   for (int icol=0; icol<nvalid_columns; icol++) {
      if (!strcmp(cfg_column.value, valid_columns[icol])) {
         colum_is_valid = true;
         break;
      }
   }

   if (!colum_is_valid) {
      fprintf(fp, "\"%s\" is not a valid option for sort_table.column\n",
              cfg_column.value);
      fprintf(fp, "Valid options are: \"%s\"",
              valid_columns[0]);
      for (int icol=1; icol<nvalid_columns; icol++) {
         fprintf(fp, ", \"%s\"", valid_columns[icol]);
      }
      fprintf(fp, "\n");
      vftr_abort(0);
   }
}

void vftr_config_assending_assert(FILE *fp, config_bool_t cfg_assending) {
   (void) fp;
   (void) cfg_assending;
}

void vftr_config_sort_table_assert(FILE *fp, config_sort_table_t cfg_sort_table,
                                   int nvalid_columns,
                                   const char *valid_columns[]) {
   vftr_config_column_assert(fp, cfg_sort_table.column,
                             nvalid_columns, valid_columns);
   vftr_config_assending_assert(fp, cfg_sort_table.ascending);
}

// Specific assertion routines
void vftr_config_off_assert(FILE *fp, config_bool_t cfg_off) {
   (void) fp;
   (void) cfg_off;
}

void vftr_config_output_directory_assert(FILE *fp,
                                         config_string_t cfg_output_directory) {
   if (cfg_output_directory.value == NULL) {
      fprintf(fp, "output_directory cannot be \"null\"\n");
      vftr_abort(0);
   }
   
   struct stat sb;
   if (stat(cfg_output_directory.value, &sb) != 0) {
      perror(cfg_output_directory.value);
      vftr_abort(0);
   }
   if (!S_ISDIR(sb.st_mode)) {
      fprintf(fp, "%s is not a valid directory\n",
              cfg_output_directory.value);
      vftr_abort(0);
   }
}

void vftr_config_outfile_basename_assert(FILE *fp,
                                         config_string_t cfg_outfile_basename) {
   (void) fp;
   (void) cfg_outfile_basename;
}

void vftr_config_logfile_for_ranks_assert(FILE *fp,
                                          config_string_t cfg_logfile_for_ranks) {
   const char *valid_values = "\"all\", \"none\", \"a-b\", \"a,b\", "
                              "or a mix of the latter two.";
   char *rangelist = cfg_logfile_for_ranks.value;
   if (!strcmp(rangelist, "all")) {
      ;
   } else if (!strcmp(rangelist, "none")) {
      ;
   } else if (rangelist == NULL) {
      fprintf(fp, "logfile_for_ranks cannot be \"null\"\n"
              "Valid values are: %s\n",
              valid_values);
      vftr_abort(0);
   } else {
      int nvals = 0;
      int *exp_list = vftr_expand_rangelist(rangelist, &nvals);
      if (nvals == 0 || exp_list == NULL) {
         fprintf(fp, "Unable to properly parse given list \"%s\" "
                 "for \"logfile_for_ranks\".\n",
                 rangelist);
         fprintf(fp, "Valid values are: %s\n",
                 valid_values);
         vftr_abort(0);
      }
      for (int i=0; i<nvals; i++) {
         if (exp_list[i] < 0) {
            fprintf(fp, "Warning: \"logfile_for_ranks\": \"%s\" "
                    "results in negative rank values.",
                    rangelist);
            fprintf(fp, "Valid values are: %s\n",
                    valid_values);
            vftr_abort(0);
         }
      }
      free(exp_list);
   }
}

void vftr_config_print_config_assert(FILE *fp, config_bool_t cfg_print_config) {
   (void) fp;
   (void) cfg_print_config;
}

void vftr_config_strip_module_names_assert(FILE *fp,
                                           config_bool_t cfg_strip_module_names) {
   (void) fp;
   (void) cfg_strip_module_names;
}

void vftr_config_demangle_cxx_assert(FILE *fp,
                                     config_bool_t cfg_demangle_cxx) {
   (void) fp;
   (void) cfg_demangle_cxx;
}

void vftr_config_include_cxx_prelude_assert(FILE *fp,
                                            config_bool_t cfg_include_cxx_prelude) {
   (void) fp;
   (void) cfg_include_cxx_prelude;
}

void vftr_config_show_calltime_imbalances_assert(FILE *fp,
                                                 config_bool_t cfg_show_callpath) {
   (void) fp;
   (void) cfg_show_callpath;
}

void vftr_config_show_callpath_assert(FILE *fp, config_bool_t cfg_show_callpath) {
   (void) fp;
   (void) cfg_show_callpath;
}

void vftr_config_show_overhead_assert(FILE *fp, config_bool_t cfg_show_overhead) {
   (void) fp;
   (void) cfg_show_overhead;
}

void vftr_config_profile_table_assert(FILE *fp,
                                      config_profile_table_t
                                      cfg_profile_table) {
   vftr_config_show_table_assert(fp, cfg_profile_table.show_table);
   vftr_config_show_calltime_imbalances_assert(fp,
      cfg_profile_table.show_calltime_imbalances);
   vftr_config_show_callpath_assert(fp, cfg_profile_table.show_callpath);
   vftr_config_show_overhead_assert(fp, cfg_profile_table.show_overhead);
   const char *valid_columns[] = {"time_excl", "time_incl", "calls",
                                  "stack_id", "overhead", "none"};
   vftr_config_sort_table_assert(fp, cfg_profile_table.sort_table,
                                 6, valid_columns);
}

void vftr_config_max_stack_ids(FILE *fp, config_int_t cfg_max_stack_ids) {
   (void) fp;
   (void) cfg_max_stack_ids;
}

void vftr_config_name_grouped_profile_table_assert(FILE *fp,
                                                   config_name_grouped_profile_table_t
                                                   cfg_profile_table) {
   vftr_config_show_table_assert(fp, cfg_profile_table.show_table);
   vftr_config_max_stack_ids(fp, cfg_profile_table.max_stack_ids);
   const char *valid_columns[] = {"time_excl", "time_incl", "calls", "none"};
   vftr_config_sort_table_assert(fp, cfg_profile_table.sort_table,
                                 4, valid_columns);
}

void vftr_config_sample_interval_assert(FILE *fp,
                                        config_float_t cfg_sampling_interval) {
   if (cfg_sampling_interval.value < 0.0) {
      fprintf(fp, "%f is not a valid value for sampling->sample_interval\n",
              cfg_sampling_interval.value);
      fprintf(fp, "Valid values are 0.0 and positive numbers\n");
      vftr_abort(0);
   }
}

void vftr_config_outbuffer_size_assert(FILE *fp,
                                       config_int_t cfg_outbuffer_size) {
   if (cfg_outbuffer_size.value <= 0) {
      fprintf(fp, "%d is not a valid value for sampling->outbuffer_size\n",
              cfg_outbuffer_size.value);
      fprintf(fp, "Valid values are positive numbers\n");
      vftr_abort(0);
   }
}

void vftr_config_precise_functions_assert(FILE *fp,
                                          config_regex_t cfg_precise_functions) {
   if (cfg_precise_functions.value != NULL && 
       cfg_precise_functions.regex == NULL) {
      fprintf(fp, "\"%s\" as argument for sampling.precise_functions "
              "could not be compiled to a valid regular expression\n",
              cfg_precise_functions.value);
      vftr_abort(0);
   }
}

void vftr_config_sampling_assert(FILE *fp,
                                 config_sampling_t cfg_sampling) {
   vftr_config_active_assert(fp, cfg_sampling.active);
   vftr_config_sample_interval_assert(fp, cfg_sampling.sample_interval);
   vftr_config_outbuffer_size_assert(fp, cfg_sampling.outbuffer_size);
   vftr_config_precise_functions_assert(fp, cfg_sampling.precise_functions);
}

void vftr_config_log_messages_assert(FILE *fp, config_bool_t cfg_log_messages) {
   (void) fp;
   (void) cfg_log_messages;
}

void vftr_config_only_for_ranks_assert(FILE *fp,
                                       config_string_t cfg_only_for_ranks) {
   const char *valid_values = "\"all\", \"none\", \"a-b\", \"a,b\", "
                              "or a mix of the latter two.";
   char *rangelist = cfg_only_for_ranks.value;
   if (!strcmp(rangelist, "all")) {
      ;
   } else if (!strcmp(rangelist, "none")) {
      ;
   } else if (rangelist == NULL) {
      fprintf(fp, "mpi.only_for_ranks cannot be \"null\"\n"
              "Valid values are: %s\n",
              valid_values);
      vftr_abort(0);
   } else {
      int nvals = 0;
      int *exp_list = vftr_expand_rangelist(rangelist, &nvals);
      if (nvals == 0 || exp_list == NULL) {
         fprintf(fp, "Unable to properly parse given list \"%s\" "
                 "for mpi.only_for_ranks.\n",
                 rangelist);
         fprintf(fp, "Valid values are: %s\n",
                 valid_values);
         vftr_abort(0);
      }
      for (int i=0; i<nvals; i++) {
         if (exp_list[i] < 0) {
            fprintf(fp, "Warning: mpi.only_for_ranks: \"%s\" "
                    "results in negative rank values.",
                    rangelist);
            fprintf(fp, "Valid values are: %s\n",
                    valid_values);
            vftr_abort(0);
         }
      }
      free(exp_list);
   }
}

void vftr_config_show_sync_time_assert(FILE *fp,
                                       config_bool_t cfg_show_sync_time) {
   (void) fp;
   (void) cfg_show_sync_time;
}

void vftr_config_mpi_assert(FILE *fp, config_mpi_t cfg_mpi) {
   vftr_config_show_table_assert(fp, cfg_mpi.show_table);
   vftr_config_log_messages_assert(fp, cfg_mpi.log_messages);
   vftr_config_only_for_ranks_assert(fp, cfg_mpi.only_for_ranks);
   vftr_config_show_sync_time_assert(fp, cfg_mpi.show_sync_time);
   vftr_config_show_callpath_assert(fp, cfg_mpi.show_callpath);
   const char *valid_columns[] = {"messages",
                                  "send_size", "recv_size",
                                  "send_bw", "recv_bw",
                                  "comm_time", "stack_id",
                                  "none"};
   vftr_config_sort_table_assert(fp, cfg_mpi.sort_table,
                                 8, valid_columns);
}

void vftr_config_cuda_assert(FILE *fp, config_cuda_t cfg_cuda) {
   vftr_config_show_table_assert(fp, cfg_cuda.show_table);
   const char *valid_columns[] = {"time", "memcpy", "cbid",
                                  "calls", "none"};
   vftr_config_sort_table_assert(fp, cfg_cuda.sort_table,
                                 5, valid_columns);
}

void vftr_config_accprof_assert(FILE *fp, config_accprof_t cfg_accprof) {
   vftr_config_show_table_assert(fp, cfg_accprof.show_table);
   vftr_config_show_table_assert(fp, cfg_accprof.show_event_details);
   const char *valid_columns[] = {"time", "memcpy", "calls", "none"};
   vftr_config_sort_table_assert(fp, cfg_accprof.sort_table, 5, valid_columns);
}

void vftr_config_hwprof_assert (FILE *fp, config_hwprof_t cfg_hwprof) {
   if (!cfg_hwprof.set) return;
   char *hwc_type = cfg_hwprof.hwc_type.value;
   if (!(!strcmp(hwc_type, "dummy") ||
         !strcmp(hwc_type, "papi")  ||
         !strcmp(hwc_type, "likwid")  ||
         !strcmp(hwc_type, "ve"))) {
      fprintf (fp, "hwprof: type is obligatory!\n");
      fprintf (fp, "Must be one of: dummy, papi, likwid, ve\n");
      fprintf (fp, "You specified: %s\n", hwc_type);
      vftr_abort(0);
   }

   int n_counters = cfg_hwprof.counters.hwc_name.n_elements;
   // Check that no builtin variable is used as a symbol name.
   for (int i = 0; i < n_counters; i++) {
     char *symbol = cfg_hwprof.counters.symbol.values[i];
     for (int j = 0; j < NSYM_BUILTIN; j++) {
        if (!strcmp(symbol, vftr_builtin_obs_symbols[j])) {
           fprintf (stderr, "hwprof: Symbol %s is reserved for a builtin variable.\n", symbol);
           vftr_abort(0);
        }
     }
   } 
#ifdef _ON_VE
   // Check that the counter matches a VE counter.
   for (int i = 0; i < n_counters; i++) {
     char *hwc_name = cfg_hwprof.counters.hwc_name.values[i];
     if (vftr_ve_counter_index (hwc_name) < 0) {
        fprintf (stderr, "hwprof: Counter %s is not a VE counter.\n", hwc_name); 
        vftr_abort(0);
     }
   }
#endif

   int n_obs = cfg_hwprof.observables.obs_name.n_elements;
   int n_formulas = cfg_hwprof.observables.formula_expr.n_elements;
   int n_units = cfg_hwprof.observables.unit.n_elements;
   int n_max = n_obs > n_formulas ? n_obs : n_formulas;
   n_max = n_units > n_max ? n_units : n_max;

   // Check that the sort column is valid
   if (n_max > 0) {
     if (cfg_hwprof.sort_by_column.value < 0 || cfg_hwprof.sort_by_column.value >= n_max) {
        fprintf (fp, "hwprof: Invalid column to sort specified: %d\n", cfg_hwprof.sort_by_column.value);
        vftr_abort(0);
     }
   } // If no observables are found, the value is irrelevant.

}

void vftr_config_assert(FILE *fp, config_t config) {
   vftr_config_off_assert(fp, config.off);
   vftr_config_output_directory_assert(fp, config.output_directory);
   vftr_config_outfile_basename_assert(fp, config.outfile_basename);
   vftr_config_logfile_for_ranks_assert(fp, config.logfile_for_ranks);
   vftr_config_print_config_assert(fp, config.print_config);
   vftr_config_strip_module_names_assert(fp, config.strip_module_names);
   vftr_config_demangle_cxx_assert(fp, config.demangle_cxx);
   vftr_config_include_cxx_prelude_assert(fp, config.include_cxx_prelude);
   vftr_config_profile_table_assert(fp, config.profile_table);
   vftr_config_name_grouped_profile_table_assert(fp, config.name_grouped_profile_table);
   vftr_config_sampling_assert(fp, config.sampling);
   vftr_config_mpi_assert(fp, config.mpi);
   vftr_config_cuda_assert(fp, config.cuda);
   vftr_config_accprof_assert(fp, config.accprof);
   vftr_config_hwprof_assert(fp, config.hwprof);
}
