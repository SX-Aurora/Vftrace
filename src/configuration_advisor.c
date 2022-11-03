#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "configuration_types.h"
#include "cJSON.h"

void vftr_print_json_config(FILE *fp, cJSON *config_json_ptr, int level) {
   for (int ilevel=0; ilevel<level; ilevel++) {
      fprintf(fp, "   ");
   }
   fprintf(fp, "%s\n", config_json_ptr->string);
   if (config_json_ptr->child != NULL) {
      vftr_print_json_config(fp, config_json_ptr->child, level+1);
   }
   if (config_json_ptr->next != NULL) {
      vftr_print_json_config(fp, config_json_ptr->next, level);
   }
}

void vftr_config_advisor(config_t *config_ptr, cJSON *config_json_ptr) {

//   vftr_print_json_config(stderr, config_json_ptr, 0);

}
