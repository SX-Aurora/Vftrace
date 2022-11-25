#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#include <string.h>

#include "configuration_types.h"
#include "configuration_advisor.h"
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
         abort();
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
         abort();
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
         abort();
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
         abort();
      }
   }
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
         abort();
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

void vftr_parse_config_hardware_scenarios(cJSON *parent_object,
                                          config_hardware_scenarios_t
                                          *cfg_hardware_scenarios_ptr) {
   // check if the Parent object has the profile table
   bool has_object = cJSON_HasObjectItem(parent_object,
                                         cfg_hardware_scenarios_ptr->name);
   if (has_object) {
      cJSON *json_object = cJSON_GetObjectItem(parent_object,
                                               cfg_hardware_scenarios_ptr->name);
      // get the child objects
      vftr_parse_config_bool(json_object,
                             &(cfg_hardware_scenarios_ptr->active));
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
      abort();
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
   vftr_parse_config_hardware_scenarios(config_json, &(config_ptr->hardware_scenarios));

   vftr_config_advisor(config_json);

   cJSON_Delete(config_json);
}
