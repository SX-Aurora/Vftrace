#include <stdlib.h>

#include "self_profile.h"
#include "configuration_types.h"
#include "configuration.h"
#include "configuration_parse.h"
#include "misc_utils.h"
#include "cJSON.h"

#ifdef _MPI
#include <mpi.h>
#endif


int main(int argc, char **argv) {
   INIT_SELF_PROF_VFTRACE;
#if defined(_MPI)
   PMPI_Init(&argc, &argv);
#else
   (void) argc;
   (void) argv;
#endif
   char *config_path = vftr_read_environment_vftr_config();
   char *config_string = vftr_read_file_to_string(config_path);
   cJSON *config_json = cJSON_Parse(config_string);
   vftr_parse_config_check_json_format(config_string);
   free(config_string);
   cJSON_Delete(config_json);

#ifdef _MPI
   PMPI_Finalize();
#endif
   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
