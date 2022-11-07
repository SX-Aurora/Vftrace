#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "self_profile.h"
#include "environment_types.h"
#include "vftrace_state.h"

#include "filenames.h"
#include "environment.h"
#include "logfile_header.h"
#include "logfile_prof_table.h"
#include "logfile_mpi_table.h"
#include "logfile_stacklist.h"
#include "collate_stacks.h"
#include "search.h"
#include "range_expand.h"
#ifdef _CUDA
#include "cuda_logfile.h"
#endif
#ifdef _ACCPROF
#include "accprof_logfile.h"
#endif

char *vftr_get_logfile_name(environment_t environment) {
   char *filename_base = vftr_create_filename_base(environment, -1, 1);
   int filename_base_len = strlen(filename_base);

   char *extension = ".log";
   int extension_len = strlen(extension);

   // construct logfile name
   int total_len = filename_base_len +
                   extension_len +
                   1; // null terminator
   char *logfile_name = (char*) malloc(total_len*sizeof(char));
   strcpy(logfile_name, filename_base);
   strcat(logfile_name, extension);

   free(filename_base);
   return logfile_name;
}

FILE *vftr_open_logfile(char *filename) {
   FILE *fp = fopen(filename, "w");
   if (fp == NULL) {
      perror(filename);
      abort();
   }
   return fp;
}

void vftr_write_logfile(vftrace_t vftrace, long long runtime) {
   SELF_PROFILE_START_FUNCTION;
   // only process 0 writes the summary logfile
   if (vftrace.process.processID != 0) {
      SELF_PROFILE_END_FUNCTION;
      return;
   }

   char *logfilename = vftr_get_logfile_name(vftrace.environment);
   FILE *fp = vftr_open_logfile(logfilename);

   vftr_write_logfile_header(fp, vftrace.timestrings);

   vftr_write_logfile_summary(fp, vftrace.process,
                              vftrace.size, runtime);

   vftr_write_logfile_profile_table(fp, vftrace.process.collated_stacktree,
                                    vftrace.environment);
   // print the name grouped profile_table
   if (vftrace.environment.group_functions_by_name.value.bool_val) {
      collated_stacktree_t namegrouped_collated_stacktree =
         vftr_collated_stacktree_group_by_name(&vftrace.process.collated_stacktree);
      vftr_write_logfile_name_grouped_profile_table(fp, namegrouped_collated_stacktree,
                                                    vftrace.environment);
      vftr_collated_stacktree_free(&namegrouped_collated_stacktree);
   }

#ifdef _MPI
   vftr_write_logfile_mpi_table(fp, vftrace.process.collated_stacktree,
                                vftrace.environment);
#endif

#ifdef _CUDA
   if (vftrace.cuda_state.n_devices == 0) {
      fprintf (fp, "The CUpti interface is enabled, but no GPU devices were found.\n");
   } else {
      vftr_write_logfile_cuda_table(fp, vftrace.process.collated_stacktree, vftrace.environment);
   }
#endif

#ifdef _ACCPROF
   vftr_write_logfile_accprof_table (fp, vftrace.process.collated_stacktree, vftrace.environment);
#endif

   vftr_write_logfile_global_stack_list(fp, vftrace.process.collated_stacktree);

   // print environment info
   vftr_print_environment(fp, vftrace.environment);

#ifdef _CUDA
   vftr_write_logfile_cbid_names (fp, vftrace.process.collated_stacktree);
#endif

   fclose(fp);
   free(logfilename);
   SELF_PROFILE_END_FUNCTION;
}
