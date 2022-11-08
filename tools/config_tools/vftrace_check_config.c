#include <stdio.h>

#include "configuration_types.h"
#include "configuration.h"
#include "configuration_defaults.h"
#include "configuration_parse.h"
#include "configuration_assert.h"
#include "misc_utils.h"

int main(int argc, char **argv) {

   if (argc < 2) {
      fprintf(stderr, "./vftrace_check_config <config.json>");
      return 1;
   }

   config_t config = vftr_set_config_default();
   char *config_string = vftr_read_file_to_string(argv[1]);
   vftr_parse_config(config_string, &config);
   free(config_string);
   vftr_config_assert(stderr, config);
   vftr_config_free(&config);

   fprintf(stdout, "Checking of config file %s is complete\n", argv[1]);
   return 0;
}
