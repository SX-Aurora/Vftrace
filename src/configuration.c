#include <stdlib.h>
#include <stdbool.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

#include "configuration_types.h"
#include "configuration_defaults.h"
#include "configuration_parse.h"
#include "bool_strings.h"
#include "misc_utils.h"

bool vftr_read_environment_vftr_off() {
   char *value = getenv("VFTR_OFF");
   if (value != NULL) {
      return vftr_string_to_bool(value);
   } else {
      return false;
   }
}

char *vftr_read_environment_vftr_config() {
   char *value = getenv("VFTR_CONFIG");
   return value;
}

config_t vftr_read_config() {
   config_t config = vftr_set_config_default();
   bool vftr_off = vftr_read_environment_vftr_off();
   if (vftr_off) {
      config.off.value = true;
   } else {
      char *config_path = vftr_read_environment_vftr_config();
      if (config_path != NULL) {
         char *config_string = vftr_read_file_to_string(config_path);
         vftr_parse_config(config_string, &config);
         free(config_string);
         config.config_file_path = strdup(config_path);
      }
   }
   return config;
}

void vftr_config_bool_free(config_bool_t *cfg_bool_ptr) {
   free(cfg_bool_ptr->name);
   cfg_bool_ptr->name = NULL;
}

void vftr_config_int_free(config_int_t *cfg_int_ptr) {
   free(cfg_int_ptr->name);
   cfg_int_ptr->name = NULL;
}

void vftr_config_float_free(config_float_t *cfg_float_ptr) {
   free(cfg_float_ptr->name);
   cfg_float_ptr->name = NULL;
}

void vftr_config_string_free(config_string_t *cfg_string_ptr) {
   free(cfg_string_ptr->name);
   cfg_string_ptr->name = NULL;
   if (cfg_string_ptr->value != NULL) {
      free(cfg_string_ptr->value);
      cfg_string_ptr->value = NULL;
   }
}

void vftr_config_string_list_free (config_string_list_t *cfg_string_list_ptr) {
   free(cfg_string_list_ptr->name);
   cfg_string_list_ptr->name = NULL;
   for (int i = 0; i < cfg_string_list_ptr->n_elements; i++) {
      free(cfg_string_list_ptr->values[i]);
   }
   free(cfg_string_list_ptr->values);
   cfg_string_list_ptr->values = NULL;
   free(cfg_string_list_ptr->list_idx);
   cfg_string_list_ptr->list_idx = NULL;
}

void vftr_config_regex_free(config_regex_t *cfg_regex_ptr) {
   free(cfg_regex_ptr->name);
   cfg_regex_ptr->name = NULL;
   if (cfg_regex_ptr->value != NULL) {
      free(cfg_regex_ptr->value);
      cfg_regex_ptr->value = NULL;
   }
   if (cfg_regex_ptr->regex != NULL) {
      regfree(cfg_regex_ptr->regex);
      cfg_regex_ptr->regex = NULL;
   }
}

void vftr_config_sort_table_free(config_sort_table_t *cfg_sort_table_ptr) {
   free(cfg_sort_table_ptr->name);
   cfg_sort_table_ptr->name = NULL;
   vftr_config_string_free(&(cfg_sort_table_ptr->column));
   vftr_config_bool_free(&(cfg_sort_table_ptr->ascending));
}

void vftr_config_profile_table_free(config_profile_table_t *cfg_profile_table_ptr) {
   free(cfg_profile_table_ptr->name);
   cfg_profile_table_ptr->name = NULL;
   vftr_config_bool_free(&(cfg_profile_table_ptr->show_table));
   vftr_config_bool_free(&(cfg_profile_table_ptr->show_calltime_imbalances));
   vftr_config_bool_free(&(cfg_profile_table_ptr->show_callpath));
   vftr_config_bool_free(&(cfg_profile_table_ptr->show_overhead));
   vftr_config_bool_free(&(cfg_profile_table_ptr->show_minmax_summary));
   vftr_config_bool_free(&(cfg_profile_table_ptr->separate));
   vftr_config_sort_table_free(&(cfg_profile_table_ptr->sort_table));
}

void vftr_config_stacklist_free(config_stacklist_t *cfg_stacklist_ptr) {
    free(cfg_stacklist_ptr->name);
    cfg_stacklist_ptr->name = NULL;
    vftr_config_bool_free(&(cfg_stacklist_ptr->show_stacklist));
}

void vftr_config_name_grouped_profile_table_free(config_name_grouped_profile_table_t
                                                 *cfg_profile_table_ptr) {
   free(cfg_profile_table_ptr->name);
   cfg_profile_table_ptr->name = NULL;
   vftr_config_bool_free(&(cfg_profile_table_ptr->show_table));
   vftr_config_int_free(&(cfg_profile_table_ptr->max_stack_ids));
   vftr_config_sort_table_free(&(cfg_profile_table_ptr->sort_table));
}

void vftr_config_sampling_free(config_sampling_t *cfg_sampling_ptr) {
   free(cfg_sampling_ptr->name);
   cfg_sampling_ptr->name = NULL;
   vftr_config_bool_free(&(cfg_sampling_ptr->active));
   vftr_config_float_free(&(cfg_sampling_ptr->sample_interval));
   vftr_config_int_free(&(cfg_sampling_ptr->outbuffer_size));
   vftr_config_regex_free(&(cfg_sampling_ptr->precise_functions));
}

void vftr_config_mpi_free(config_mpi_t *cfg_mpi_ptr) {
   free(cfg_mpi_ptr->name);
   cfg_mpi_ptr->name = NULL;
   vftr_config_bool_free(&(cfg_mpi_ptr->show_table));
   vftr_config_bool_free(&(cfg_mpi_ptr->log_messages));
   vftr_config_string_free(&(cfg_mpi_ptr->only_for_ranks));
   vftr_config_bool_free(&(cfg_mpi_ptr->show_sync_time));
   vftr_config_bool_free(&(cfg_mpi_ptr->show_callpath));
   vftr_config_sort_table_free(&(cfg_mpi_ptr->sort_table));
}

void vftr_config_cuda_free(config_cuda_t *cfg_cuda_ptr) {
   free(cfg_cuda_ptr->name);
   cfg_cuda_ptr->name = NULL;
   vftr_config_bool_free(&(cfg_cuda_ptr->show_table));
   vftr_config_sort_table_free(&(cfg_cuda_ptr->sort_table));
}

void vftr_config_accprof_free(config_accprof_t *cfg_accprof_ptr) {
   free(cfg_accprof_ptr->name);
   cfg_accprof_ptr->name = NULL;
   vftr_config_bool_free(&(cfg_accprof_ptr->active));
   vftr_config_bool_free(&(cfg_accprof_ptr->show_table));
   vftr_config_bool_free(&(cfg_accprof_ptr->show_event_details));
   vftr_config_sort_table_free(&(cfg_accprof_ptr->sort_table));
}

void  vftr_config_hwcounters_free (config_hwcounters_t *cfg_hwc_ptr) {
   free(cfg_hwc_ptr->name);
   cfg_hwc_ptr->name = NULL;
   vftr_config_string_list_free(&(cfg_hwc_ptr->hwc_name));
   vftr_config_string_list_free(&(cfg_hwc_ptr->symbol));
}

void  vftr_config_hwobservables_free (config_hwobservables_t *cfg_hwobs_ptr) {
   free(cfg_hwobs_ptr->name);
   cfg_hwobs_ptr->name = NULL;
   if (cfg_hwobs_ptr->obs_name.set) vftr_config_string_list_free(&(cfg_hwobs_ptr->obs_name));
   if (cfg_hwobs_ptr->formula_expr.set) vftr_config_string_list_free(&(cfg_hwobs_ptr->formula_expr));
   if (cfg_hwobs_ptr->unit.set) vftr_config_string_list_free(&(cfg_hwobs_ptr->unit));
}

void vftr_config_hwprof_free (config_hwprof_t *cfg_hwprof_ptr) {
   free(cfg_hwprof_ptr->name);
   cfg_hwprof_ptr->name = NULL;
   vftr_config_hwcounters_free (&(cfg_hwprof_ptr->counters));
   vftr_config_hwobservables_free (&(cfg_hwprof_ptr->observables));
}

void vftr_config_free(config_t *config_ptr) {
   vftr_config_bool_free(&(config_ptr->off));
   vftr_config_string_free(&(config_ptr->output_directory));
   vftr_config_string_free(&(config_ptr->outfile_basename));
   vftr_config_string_free(&(config_ptr->logfile_for_ranks));
   vftr_config_bool_free(&(config_ptr->print_config));
   vftr_config_bool_free(&(config_ptr->strip_module_names));
   vftr_config_bool_free(&(config_ptr->demangle_cxx));
   vftr_config_bool_free(&(config_ptr->include_cxx_prelude));
   vftr_config_profile_table_free(&(config_ptr->profile_table));
   vftr_config_stacklist_free(&(config_ptr->stacklist));
   vftr_config_name_grouped_profile_table_free(&(config_ptr->name_grouped_profile_table));
   vftr_config_sampling_free(&(config_ptr->sampling));
   vftr_config_mpi_free(&(config_ptr->mpi));
   vftr_config_cuda_free(&(config_ptr->cuda));
   vftr_config_accprof_free(&(config_ptr->accprof));
   vftr_config_hwprof_free(&(config_ptr->hwprof));

   if (config_ptr->config_file_path != NULL) {
      free(config_ptr->config_file_path);
      config_ptr->config_file_path = NULL;
   }
   config_ptr->valid = false;
}
