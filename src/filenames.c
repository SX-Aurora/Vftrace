#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <libgen.h>

#include "configuration_types.h"

#include "self_profile.h"
#include "misc_utils.h"
#include "exe_info.h"

char *vftr_create_filename_base(config_t config, int rankID, int nranks) {
   SELF_PROFILE_START_FUNCTION;
   char *out_dir = NULL;
   out_dir = strdup(config.output_directory.value);
   int out_dir_len = strlen(out_dir);

   char *exe_name = NULL;
   if (config.outfile_basename.set && config.outfile_basename.value != NULL) {
      // user defined logfile name
      exe_name = strdup(config.outfile_basename.value);
   } else {
      // default name constructed from executable name
      char *exe_path = vftr_get_executable_path();
      if (exe_path == NULL) {
         exe_name = strdup("unknown");
      } else {
         // name will be <exe_name>_<rankID>.log
         // exe_name:
         exe_name = strdup(basename(exe_path));
      }
      free(exe_path);
   }
   int exe_name_len = strlen(exe_name);

   // rankID (leading zeroes for nice sorting among ranks):
   int ndigits = 0;
   int rankID_len = 0;
   char *allstr = "all";
   if (rankID < 0) {
      rankID_len = strlen(allstr);
   } else {
      ndigits = vftr_count_base_digits(nranks, 10);
      rankID_len = snprintf(NULL, 0, "%0*d", ndigits, rankID);
   }

   // construct filename base
   int total_len = out_dir_len +
                   1 + // folder separating slash
                   exe_name_len +
                   1 + // underscore
                   rankID_len +
                   1; // null terminator
   char *filename_base = (char*) malloc(total_len*sizeof(char));
   strcpy(filename_base, out_dir);
   strcat(filename_base, "/");
   strcat(filename_base, exe_name);
   filename_base[out_dir_len+1+exe_name_len] = '_';
   if (rankID < 0) {
      snprintf(out_dir_len+1+filename_base+exe_name_len+1, rankID_len+1,
               "%*s", ndigits, allstr);
   } else {
      snprintf(out_dir_len+1+filename_base+exe_name_len+1, rankID_len+1,
               "%0*d", ndigits, rankID);
   }

   free(exe_name);
   free(out_dir);
   SELF_PROFILE_END_FUNCTION;
   return filename_base;
}
