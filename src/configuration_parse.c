#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#include <string.h>

#include "misc_utils.h"
#include "signal_handling.h"
#include "configuration_types.h"
#include "configuration_advisor.h"
#include "configuration_parse.h"
#include "regular_expressions.h"
#include "range_expand.h"
#include "cJSON.h"

void vftr_parse_config_bool(cJSON *parent_object, config_bool_t *cfg_bool_ptr) {
   // check if the Parent object has the boolean
   bool has_object = cJSON_HasObjectItem(parent_object,
                                         cfg_bool_ptr->name);
   if (has_object) {
      cJSON *json_object = cJSON_GetObjectItem(parent_object,
                                               cfg_bool_ptr->name);
      // test if it is a valid bool
      if (cJSON_IsBool(json_object)) {
         cfg_bool_ptr->value = cJSON_IsTrue(json_object);
         cfg_bool_ptr->set = true;
      } else {
         fprintf(stderr, "%s->%s expects a boolean.\n",
                 parent_object->string, json_object->string);
         vftr_abort(0);
      }
   }
}

void vftr_parse_config_int(cJSON *parent_object, config_int_t *cfg_int_ptr) {
   // check if the Parent object has the int
   bool has_object = cJSON_HasObjectItem(parent_object,
                                         cfg_int_ptr->name);
   if (has_object) {
      cJSON *json_object = cJSON_GetObjectItem(parent_object,
                                               cfg_int_ptr->name);
      // test if it is a valid number
      if (cJSON_IsNumber(json_object)) {
         cfg_int_ptr->value = (int) cJSON_GetNumberValue(json_object);
         cfg_int_ptr->set = true;
      } else {
         fprintf(stderr, "%s->%s expects an integer.\n",
                 parent_object->string, json_object->string);
         vftr_abort(0);
      }
   }
}

void vftr_parse_config_float(cJSON *parent_object, config_float_t *cfg_float_ptr) {
   // check if the Parent object has the float
   bool has_object = cJSON_HasObjectItem(parent_object,
                                         cfg_float_ptr->name);
   if (has_object) {
      cJSON *json_object = cJSON_GetObjectItem(parent_object,
                                               cfg_float_ptr->name);
      // test if it is a valid number
      if (cJSON_IsNumber(json_object)) {
         cfg_float_ptr->value = cJSON_GetNumberValue(json_object);
         cfg_float_ptr->set = true;
      } else {
         fprintf(stderr, "%s->%s expects an float.\n",
                 parent_object->string, json_object->string);
         vftr_abort(0);
      }
   }
}

void vftr_parse_config_string(cJSON *parent_object, config_string_t *cfg_string_ptr) {
   // check if the Parent object has the string
   bool has_object = cJSON_HasObjectItem(parent_object,
                                         cfg_string_ptr->name);
   if (has_object) {
      cJSON *json_object = cJSON_GetObjectItem(parent_object,
                                               cfg_string_ptr->name);
      // test if it is a valid number
      if (cJSON_IsString(json_object)) {
         if (cfg_string_ptr->value != NULL) {
            free(cfg_string_ptr->value);
         }
         cfg_string_ptr->value = strdup(cJSON_GetStringValue(json_object));
         cfg_string_ptr->set = true;
      } else if (cJSON_IsNull(json_object)) {
         if (cfg_string_ptr->value != NULL) {
            free(cfg_string_ptr->value);
         }
         cfg_string_ptr->value = NULL;
         cfg_string_ptr->set = true;
      } else {
         fprintf(stderr, "%s->%s expects a string of null.\n",
                 parent_object->string, json_object->string);
         vftr_abort(0);
      }
   }
}

void vftr_parse_config_string_list (cJSON *parent_object, char *list_name, config_string_list_t *cfg_string_list_ptr) {
   cfg_string_list_ptr->n_elements = 0;
   bool has_object = cJSON_HasObjectItem (parent_object, list_name);
   if (!has_object) return;
   int idx, n;
   for (int pass = 0; pass < 2; pass++) {
      n = 0;
      idx = 0;
      cJSON *json_list = cJSON_GetObjectItem(parent_object, list_name);
      cJSON *json_object;
      cJSON_ArrayForEach(json_object, json_list) {
         cJSON *token = cJSON_GetObjectItem(json_object, cfg_string_list_ptr->name);
         if (token != NULL && pass == 1) {
            cfg_string_list_ptr->values[n] = strdup(cJSON_GetStringValue(token));
            cfg_string_list_ptr->list_idx[n] = idx;
         }
         if (token != NULL) n++;
         idx++;
      }
      if (pass == 0) {
         cfg_string_list_ptr->n_elements = n;
         cfg_string_list_ptr->values = (char**)malloc(n * sizeof(char*));
         cfg_string_list_ptr->list_idx = (int*)malloc(n * sizeof(int));
         for (int i = 0; i < n; i++) {
            cfg_string_list_ptr->values[i] = NULL;
            cfg_string_list_ptr->list_idx[i] = -1;
         }
      }
   } 
   cfg_string_list_ptr->set = true;
}

void vftr_parse_config_regex(cJSON *parent_object, config_regex_t *cfg_regex_ptr) {
   // check if the Parent object has the regex
   bool has_object = cJSON_HasObjectItem(parent_object,
                                         cfg_regex_ptr->name);
   if (has_object) {
      cJSON *json_object = cJSON_GetObjectItem(parent_object,
                                               cfg_regex_ptr->name);
      // test if it is a valid number
      if (cJSON_IsString(json_object)) {
         if (cfg_regex_ptr->value != NULL) {
            free(cfg_regex_ptr->value);
         }  
         cfg_regex_ptr->value = strdup(cJSON_GetStringValue(json_object));
         cfg_regex_ptr->regex = vftr_compile_regexp(cfg_regex_ptr->value);
         cfg_regex_ptr->set = true;
      } else if (cJSON_IsNull(json_object)) {
         if (cfg_regex_ptr->value != NULL) {
            free(cfg_regex_ptr->value);
         }
         cfg_regex_ptr->value = NULL;
         cfg_regex_ptr->set = true;
      } else {
         fprintf(stderr, "%s->%s expects a string of null.\n",
                 parent_object->string, json_object->string);
         vftr_abort(0);
      }
   }
}

void vftr_parse_config_sort_table(cJSON *parent_object,
                                  config_sort_table_t *cfg_sort_table_ptr) {
   // check if the Parent object has the sort table
   bool has_object = cJSON_HasObjectItem(parent_object,
                                         cfg_sort_table_ptr->name);
   if (has_object) {
      cJSON *json_object = cJSON_GetObjectItem(parent_object,
                                               cfg_sort_table_ptr->name);
      // get the child objects
      vftr_parse_config_string(json_object, &(cfg_sort_table_ptr->column));
      vftr_parse_config_bool(json_object, &(cfg_sort_table_ptr->ascending));
   }
}

void vftr_parse_config_profile_table(cJSON *parent_object,
                                     config_profile_table_t *cfg_profile_table_ptr) {
   // check if the Parent object has the profile table
   bool has_object = cJSON_HasObjectItem(parent_object,
                                         cfg_profile_table_ptr->name);
   if (has_object) {
      cJSON *json_object = cJSON_GetObjectItem(parent_object,
                                               cfg_profile_table_ptr->name);
      // get the child objects
      vftr_parse_config_bool(json_object,
                             &(cfg_profile_table_ptr->show_table));
      vftr_parse_config_bool(json_object,
                             &(cfg_profile_table_ptr->show_calltime_imbalances));
      vftr_parse_config_bool(json_object,
                             &(cfg_profile_table_ptr->show_callpath));
      vftr_parse_config_bool(json_object,
                             &(cfg_profile_table_ptr->show_overhead));
      vftr_parse_config_bool(json_object,
                             &(cfg_profile_table_ptr->show_minmax_summary));
      vftr_parse_config_bool(json_object,
                             &(cfg_profile_table_ptr->separate));
      vftr_parse_config_sort_table(json_object,
                                   &(cfg_profile_table_ptr->sort_table));
   }
}

void vftr_parse_config_name_grouped_profile_table(cJSON *parent_object,
                                                  config_name_grouped_profile_table_t
                                                  *cfg_profile_table_ptr) {
   // check if the Parent object has the profile table
   bool has_object = cJSON_HasObjectItem(parent_object,
                                         cfg_profile_table_ptr->name);
   if (has_object) {
      cJSON *json_object = cJSON_GetObjectItem(parent_object,
                                               cfg_profile_table_ptr->name);
      // get the child objects
      vftr_parse_config_bool(json_object,
                             &(cfg_profile_table_ptr->show_table));
      vftr_parse_config_int(json_object,
                            &(cfg_profile_table_ptr->max_stack_ids));
      vftr_parse_config_sort_table(json_object,
                                   &(cfg_profile_table_ptr->sort_table));
   }
}

void vftr_parse_config_sampling(cJSON *parent_object,
                                config_sampling_t *cfg_sampling_ptr) {
   // check if the Parent object has the profile table
   bool has_object = cJSON_HasObjectItem(parent_object,
                                         cfg_sampling_ptr->name);
   if (has_object) {
      cJSON *json_object = cJSON_GetObjectItem(parent_object,
                                               cfg_sampling_ptr->name);
      // get the child objects
      vftr_parse_config_bool(json_object,
                             &(cfg_sampling_ptr->active));
      vftr_parse_config_float(json_object,
                              &(cfg_sampling_ptr->sample_interval));
      vftr_parse_config_int(json_object,
                            &(cfg_sampling_ptr->outbuffer_size));
      vftr_parse_config_regex(json_object,
                              &(cfg_sampling_ptr->precise_functions));
   }
}

void vftr_parse_config_mpi(cJSON *parent_object,
                           config_mpi_t *cfg_mpi_ptr) {
   // check if the Parent object has the profile table
   bool has_object = cJSON_HasObjectItem(parent_object,
                                         cfg_mpi_ptr->name);
   if (has_object) {
      cJSON *json_object = cJSON_GetObjectItem(parent_object,
                                               cfg_mpi_ptr->name);
      // get the child objects
      vftr_parse_config_bool(json_object,
                             &(cfg_mpi_ptr->show_table));
      vftr_parse_config_bool(json_object,
                             &(cfg_mpi_ptr->log_messages));
      vftr_parse_config_string(json_object,
                               &(cfg_mpi_ptr->only_for_ranks));
      vftr_parse_config_bool(json_object,
                             &(cfg_mpi_ptr->show_sync_time));
      vftr_parse_config_bool(json_object,
                             &(cfg_mpi_ptr->show_callpath));
      vftr_parse_config_sort_table(json_object,
                                   &(cfg_mpi_ptr->sort_table));
   }
}

void vftr_parse_config_cuda(cJSON *parent_object,
                            config_cuda_t *cfg_cuda_ptr) {
   // check if the Parent object has the profile table
   bool has_object = cJSON_HasObjectItem(parent_object, cfg_cuda_ptr->name);
   if (has_object) {
      cJSON *json_object = cJSON_GetObjectItem(parent_object, cfg_cuda_ptr->name);
      // get the child objects
      vftr_parse_config_bool(json_object, &(cfg_cuda_ptr->show_table));
      vftr_parse_config_sort_table(json_object, &(cfg_cuda_ptr->sort_table));
   }
}

void vftr_parse_config_accprof(cJSON *parent_object, config_accprof_t *cfg_accprof_ptr) {
   // check if the Parent object has the profile table
   bool has_object = cJSON_HasObjectItem(parent_object, cfg_accprof_ptr->name);
   if (has_object) {
      cJSON *json_object = cJSON_GetObjectItem(parent_object, cfg_accprof_ptr->name);
      // get the child objects
      vftr_parse_config_bool(json_object, &(cfg_accprof_ptr->show_table));
      vftr_parse_config_bool(json_object, &(cfg_accprof_ptr->show_event_details));
      vftr_parse_config_sort_table(json_object, &(cfg_accprof_ptr->sort_table));
   }
}

void vftr_parse_config_hwcounters (cJSON *parent_object, config_hwcounters_t *cfg_hwc) {
   bool has_object = cJSON_HasObjectItem(parent_object, cfg_hwc->name);
   if (!has_object) return;
   vftr_parse_config_string_list (parent_object, cfg_hwc->name, &(cfg_hwc->hwc_name));
   vftr_parse_config_string_list (parent_object, cfg_hwc->name, &(cfg_hwc->symbol));
   cfg_hwc->set = cfg_hwc->hwc_name.n_elements > 0 && cfg_hwc->symbol.n_elements > 0;
}

void vftr_parse_config_hwobservables (cJSON *parent_object, config_hwobservables_t *cfg_hwobs) {
   bool has_object = cJSON_HasObjectItem(parent_object, cfg_hwobs->name);
   if (!has_object) return;
   vftr_parse_config_string_list (parent_object, cfg_hwobs->name, &(cfg_hwobs->obs_name));
   vftr_parse_config_string_list (parent_object, cfg_hwobs->name, &(cfg_hwobs->formula_expr));
   vftr_parse_config_string_list (parent_object, cfg_hwobs->name, &(cfg_hwobs->unit));
   cfg_hwobs->set = cfg_hwobs->obs_name.n_elements > 0;
}

void vftr_parse_config_hwprof (cJSON *parent_object, config_hwprof_t *cfg_hwprof) {
   bool has_object = cJSON_HasObjectItem(parent_object, cfg_hwprof->name);
   if (has_object) {
      cJSON *json_object = cJSON_GetObjectItem (parent_object, cfg_hwprof->name);
      vftr_parse_config_bool (json_object, &(cfg_hwprof->active));
      vftr_parse_config_string (json_object, &(cfg_hwprof->hwc_type));
      vftr_parse_config_bool (json_object, &(cfg_hwprof->show_observables));
      vftr_parse_config_bool (json_object, &(cfg_hwprof->show_counters));
      vftr_parse_config_bool (json_object, &(cfg_hwprof->show_summary));
      vftr_parse_config_int (json_object, &(cfg_hwprof->sort_by_column));
      vftr_parse_config_string (json_object, &(cfg_hwprof->default_scenario));
      vftr_parse_config_hwcounters(json_object, &(cfg_hwprof->counters));
      vftr_parse_config_hwobservables(json_object, &(cfg_hwprof->observables));
      // There are two ways in which hardware profiling can be defined: Inline or default scenarios. 
      // Both options exclude each other, so we abort in that case.
      bool has_default_scenario = cfg_hwprof->default_scenario.set &&
                                  strcmp(cfg_hwprof->default_scenario.value, "");
      bool has_inline_hwc_or_obs = cfg_hwprof->counters.set || cfg_hwprof->observables.set;
      if (has_default_scenario && has_inline_hwc_or_obs) {
         fprintf (stderr, "PAPI configuration has both a default scenario and inline counters and / or observables\n");
         vftr_abort(0); 
      } else if (has_default_scenario) {
         vftr_parse_config_string (json_object, &(cfg_hwprof->default_scenario));
         FILE *fp = fopen (cfg_hwprof->default_scenario.value, "r");
         if (fp == NULL) {
            fprintf (stderr, "Could not find default PAPI scenario %s\n", cfg_hwprof->default_scenario.value);
         }
         char *config_string = vftr_read_file_to_string(cfg_hwprof->default_scenario.value);
         cJSON *config_tmp = cJSON_Parse(config_string);
         free(config_string);
         fclose (fp);

         vftr_parse_config_hwcounters(config_tmp, &(cfg_hwprof->counters));
         vftr_parse_config_hwobservables(config_tmp, &(cfg_hwprof->observables));
      }
      cfg_hwprof->set = true;
   }
}

int vftr_parse_config_check_json_format(char *config_string) {
   // if parsing of the JSON-file fails this will return 
   // a pointer to the faulty position in the string
   const char *errorstr = cJSON_GetErrorPtr();
   if (errorstr != NULL) {
      // Determine line in which the the failure occoured
      int linenr = 1;
      char *tmpstr = config_string;
      while (tmpstr < errorstr) {
         if (*tmpstr == '\n') {
            linenr++;
         }
         tmpstr++;
      }
      fprintf(stderr, "Could not parse vftrace config file.\n"
              "Parsing failed in line %d\n", linenr);

      const int max_prev_lines = 3;
      const int max_foll_lines = 3;
      int prev_lines = 0;
      tmpstr = (char*) errorstr;
      while (prev_lines <= max_prev_lines && tmpstr > config_string) {
         if (*tmpstr == '\n') {
            prev_lines++;
         }
         tmpstr--;
      }
      if (tmpstr > config_string) {
         tmpstr++;
         fprintf(stderr, "...");
      }
      while (tmpstr < errorstr) {
         fputc(*tmpstr, stderr);
         tmpstr++;
      }
      fprintf(stderr, "%16s<= Parsing error occoured here\n", "");
      int foll_lines = 0;
      while (foll_lines < max_foll_lines && *tmpstr != '\0') {
         if (*tmpstr == '\n') {
            foll_lines++;
         }
         fputc(*tmpstr, stderr);
         tmpstr++;
      }
      if (*tmpstr != '\0') {
         fprintf(stderr, "...\n");
      }
      return 1;
   }

   return 0;
}

void vftr_parse_config(char *config_string, config_t *config_ptr) {
   cJSON *config_json = cJSON_Parse(config_string);
   if (vftr_parse_config_check_json_format(config_string)) {
      vftr_abort(0);
   }

   // Set the cJSON internal name for the overarching struct,
   // to aid with error messages in the tree.
   config_json->string = strdup("vftrace_config");

   vftr_parse_config_bool(config_json, &(config_ptr->off));
   vftr_parse_config_string(config_json, &(config_ptr->output_directory));
   vftr_parse_config_string(config_json, &(config_ptr->outfile_basename));
   vftr_parse_config_string(config_json, &(config_ptr->logfile_for_ranks));
   vftr_parse_config_bool(config_json, &(config_ptr->print_config));
   vftr_parse_config_bool(config_json, &(config_ptr->print_config));
   vftr_parse_config_bool(config_json, &(config_ptr->strip_module_names));
   vftr_parse_config_bool(config_json, &(config_ptr->demangle_cxx));
   vftr_parse_config_bool(config_json, &(config_ptr->include_cxx_prelude));
   vftr_parse_config_profile_table(config_json, &(config_ptr->profile_table));
   vftr_parse_config_name_grouped_profile_table(config_json,
         &(config_ptr->name_grouped_profile_table));
   vftr_parse_config_sampling(config_json, &(config_ptr->sampling));
   vftr_parse_config_mpi(config_json, &(config_ptr->mpi));
   vftr_parse_config_cuda(config_json, &(config_ptr->cuda));
   vftr_parse_config_accprof(config_json, &(config_ptr->accprof));
   vftr_parse_config_hwprof(config_json, &(config_ptr->hwprof));

   vftr_config_advisor(config_json);

   cJSON_Delete(config_json);
}
