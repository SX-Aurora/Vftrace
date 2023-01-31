#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "configuration_types.h"
#include "levenshtein.h"
#include "cJSON.h"

void vftr_config_advisor_check_options(int noptions, char *options[],
                                       cJSON *json_obj) {
   while (json_obj != NULL) {
      int minidx = 0;
      int mindist = vftr_levenshtein_distance(options[minidx], json_obj->string);
      for (int iopt=1; iopt<noptions; iopt ++) {
         int tmpdist = vftr_levenshtein_distance(options[iopt], json_obj->string);
         if (tmpdist < mindist) {
            mindist = tmpdist;
            minidx = iopt;
         }
      }
      if (mindist > 0) {
         fprintf(stderr,
                 "Warning: \"%s\" is not a valid vftrace option.\n"
                 "         Did you mean \"%s\"?\n",
                 json_obj->string, options[minidx]);
      }
      json_obj = json_obj->next;
   }
}

void vftr_config_advisor_sort_table(cJSON *json_obj) {
   char *options[] = {
      "column",
      "ascending"
   };
   int noptions = sizeof(options) / sizeof(char*);
   vftr_config_advisor_check_options(noptions, options, json_obj);
}

void vftr_config_advisor_profile_table(cJSON *json_obj) {
   char *options[] = {
      "show_table",
      "show_calltime_imbalances",
      "show_callpath",
      "show_overhead",
      "sort_table"
   };
   int noptions = sizeof(options) / sizeof(char*);
   vftr_config_advisor_check_options(noptions, options, json_obj->child);
   
   char *sec_name = NULL;
   bool has_object;

   sec_name = "sort_table";
   has_object = cJSON_HasObjectItem(json_obj, sec_name);
   if (has_object) {
      cJSON *json_sec = cJSON_GetObjectItem(json_obj, sec_name);
      vftr_config_advisor_sort_table(json_sec->child);
   }
}

void vftr_config_advisor_name_grouped_profile_table(cJSON *json_obj) {
   char *options[] = {
      "show_table",
      "max_stack_ids",
      "sort_table"
   };
   int noptions = sizeof(options) / sizeof(char*);
   vftr_config_advisor_check_options(noptions, options, json_obj->child);
 
   char *sec_name = NULL;
   bool has_object;

   sec_name = "sort_table";
   has_object = cJSON_HasObjectItem(json_obj, sec_name);
   if (has_object) {
      cJSON *json_sec = cJSON_GetObjectItem(json_obj, sec_name);
      vftr_config_advisor_sort_table(json_sec->child);
   }
}

void vftr_config_advisor_sampling(cJSON *json_obj) {
   char *options[] = {
      "active",
      "sample_interval",
      "outbuffer_size",
      "precise_functions"
   };
   int noptions = sizeof(options) / sizeof(char*);
   vftr_config_advisor_check_options(noptions, options, json_obj->child);
}

void vftr_config_advisor_mpi(cJSON *json_obj) {
   char *options[] = {
      "show_table",
      "log_messages",
      "only_for_ranks",
      "show_sync_time",
      "show_callpath",
      "sort_table"
   };
   int noptions = sizeof(options) / sizeof(char*);
   vftr_config_advisor_check_options(noptions, options, json_obj->child);

   char *sec_name = NULL;
   bool has_object;
   sec_name = "sort_table";
   has_object = cJSON_HasObjectItem(json_obj, sec_name);
   if (has_object) {
      cJSON *json_sec = cJSON_GetObjectItem(json_obj, sec_name);
      vftr_config_advisor_sort_table(json_sec->child);
   }
}

void vftr_config_advisor_cuda(cJSON *json_obj) {
   char *options[] = {
      "show_table",
      "sort_table"
   };
   int noptions = sizeof(options) / sizeof(char*);
   vftr_config_advisor_check_options(noptions, options, json_obj->child);

   char *sec_name = NULL;
   bool has_object;
   sec_name = "sort_table";
   has_object = cJSON_HasObjectItem(json_obj, sec_name);
   if (has_object) {
      cJSON *json_sec = cJSON_GetObjectItem(json_obj, sec_name);
      vftr_config_advisor_sort_table(json_sec->child);
   }
}

void vftr_config_advisor_hwprof (cJSON *json_obj) {
  char *options[] = {
     "active",
     "type",
     "show_tables",
     "show_counters",
     "show_observables",
     "show_summary",
     "sort_by_column",
     "default_scenario",
     "counters",
     "observables"
  };
  int noptions = sizeof(options) / sizeof(char*);
  vftr_config_advisor_check_options(noptions, options, json_obj->child);

  char *options_counter[] = {
     "hwc_name",
     "symbol"
  };
  int noptions_counter = sizeof(options_counter) / sizeof(char*);

  cJSON *json_array = cJSON_GetObjectItem(json_obj, "counters");
  cJSON *array_element;

  if (json_array != NULL) {
     cJSON_ArrayForEach(array_element, json_array) {
        vftr_config_advisor_check_options (noptions_counter, options_counter, array_element->child);     
     }
  }

  char *options_observables[] = {
     "name",
     "formula",
     "unit"
  };
  int noptions_observables = sizeof(options_observables) / sizeof(char*);
  
  json_array = cJSON_GetObjectItem(json_obj, "observables");
   
  if (json_array != NULL) {
     cJSON_ArrayForEach(array_element, json_array) {
        vftr_config_advisor_check_options (noptions_observables, options_observables, array_element->child);
     }
  }
}

void vftr_config_advisor(cJSON *config_json_ptr) {
   char *options[] = {
      "off",
      "output_directory",
      "outfile_basename",
      "logfile_for_ranks",
      "print_config",
      "strip_module_names",
      "demangle_cxx",
      "include_cxx_prelude",
      "profile_table",
      "name_grouped_profile_table",
      "sampling",
      "mpi",
      "cuda",
      "openacc",
      "hwprof"
   };
   int noptions = sizeof(options) / sizeof(char*);
   vftr_config_advisor_check_options(noptions, options, config_json_ptr->child);

   bool has_object;
   char *sec_name = NULL;

   sec_name = "profile_table";
   has_object = cJSON_HasObjectItem(config_json_ptr, sec_name);
   if (has_object) {
      cJSON *json_sec = cJSON_GetObjectItem(config_json_ptr, sec_name);
      vftr_config_advisor_profile_table(json_sec);
   }

   sec_name = "name_grouped_profile_table";
   has_object = cJSON_HasObjectItem(config_json_ptr, sec_name);
   if (has_object) {
      cJSON *json_sec = cJSON_GetObjectItem(config_json_ptr, sec_name);
      vftr_config_advisor_name_grouped_profile_table(json_sec);
   }

   sec_name = "sampling";
   has_object = cJSON_HasObjectItem(config_json_ptr, sec_name);
   if (has_object) {
      cJSON *json_sec = cJSON_GetObjectItem(config_json_ptr, sec_name);
      vftr_config_advisor_sampling(json_sec);
   } 

   sec_name = "mpi";
   has_object = cJSON_HasObjectItem(config_json_ptr, sec_name);
   if (has_object) {
      cJSON *json_sec = cJSON_GetObjectItem(config_json_ptr, sec_name);
      vftr_config_advisor_mpi(json_sec);
   }

   sec_name = "cuda";
   has_object = cJSON_HasObjectItem(config_json_ptr, sec_name);
   if (has_object) {
      cJSON *json_sec = cJSON_GetObjectItem(config_json_ptr, sec_name);
      vftr_config_advisor_cuda(json_sec);
   }

   sec_name = "hwprof";
   has_object = cJSON_HasObjectItem(config_json_ptr, sec_name);
   if (has_object) {
      cJSON *json_sec = cJSON_GetObjectItem(config_json_ptr, sec_name);
      vftr_config_advisor_hwprof(json_sec);
   }
}
