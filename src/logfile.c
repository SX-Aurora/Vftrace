#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "self_profile.h"
#include "configuration_types.h"
#include "vftrace_state.h"

#include "filenames.h"
#include "configuration_print.h"
#include "logfile_header.h"
#include "logfile_prof_table.h"
#ifdef _MPI
#include "logfile_mpi_table.h"
#endif
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
#ifdef _VEDA
#include "logfile_veda_table.h"
#endif

char *vftr_get_logfile_name(config_t config) {
   char *filename_base = vftr_create_filename_base(config, -1, 1);
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

   char *logfilename = vftr_get_logfile_name(vftrace.config);
   FILE *fp = vftr_open_logfile(logfilename);

   vftr_write_logfile_header(fp, vftrace.timestrings);

   vftr_write_logfile_summary(fp, vftrace.process,
                              vftrace.size, runtime);

   if (vftrace.config.profile_table.show_table.value) {
      vftr_write_logfile_profile_table(fp, vftrace.process.collated_stacktree,
                                       vftrace.config);
   }
   // print the name grouped profile_table
   if (vftrace.config.name_grouped_profile_table.show_table.value) {
      collated_stacktree_t namegrouped_collated_stacktree =
         vftr_collated_stacktree_group_by_name(&vftrace.process.collated_stacktree);
      vftr_write_logfile_name_grouped_profile_table(fp, namegrouped_collated_stacktree,
                                                    vftrace.config);
      vftr_collated_stacktree_free(&namegrouped_collated_stacktree);
   }

#ifdef _MPI
   int mpi_initialized;
   PMPI_Initialized(&mpi_initialized);
   if (vftrace.config.mpi.show_table.value && mpi_initialized) {
      vftr_write_logfile_mpi_table(fp, vftrace.process.collated_stacktree,
                                   vftrace.config);
   }
#endif

#ifdef _CUDA
   if (vftrace.cuda_state.n_devices == 0) {
      fprintf (fp, "The CUpti interface is enabled, but no GPU devices were found.\n");
   } else if (vftrace.config.cuda.show_table.value) {
      vftr_write_logfile_cuda_table(fp, vftrace.process.collated_stacktree, vftrace.config);
   }
#endif

#ifdef _ACCPROF
   if (vftrace.accprof_state.n_devices == 0) {
      fprintf (fp, "\nThe ACCProf interface is enabled, but no GPU devices were found.\n");
   } else if (!vftr_has_accprof_events (vftrace.process.collated_stacktree)) {
      fprintf (fp, "\nNo OpenACC events have been registered.\n");
   } else if (vftrace.config.accprof.show_table.value) {
      if (vftrace.accprof_state.n_open_wait_queues > 0) {
         fprintf (fp, "\nWarning: There are %d unresolved OpenACC wait regions.\n", 
                  vftrace.accprof_state.n_open_wait_queues);
      }
      vftr_write_logfile_accprof_grouped_table (fp, vftrace.process.collated_stacktree, vftrace.config);
      if (vftrace.config.accprof.show_event_details.value) {
         vftr_write_logfile_accprof_event_table (fp, vftrace.process.collated_stacktree, vftrace.config);
      }
   }
#endif

#ifdef _VEDA
   // TODO CONFIG FOR VEDA
   vftr_write_logfile_veda_table(fp, vftrace.process.collated_stacktree,
                                 vftrace.config);
#endif

   vftr_write_logfile_global_stack_list(fp, vftrace.process.collated_stacktree);

   // print config info
   if (vftrace.config.print_config.value) {
      vftr_print_config(fp, vftrace.config, true);
   }

#ifdef _CUDA
   if (vftrace.config.cuda.show_table.value) {
      vftr_write_logfile_cbid_names (fp, vftrace.process.collated_stacktree);
   }
#endif

#ifdef _ACCPROF
   if (vftrace.config.accprof.show_event_details.value) vftr_write_logfile_accev_names (fp);
#endif

   fclose(fp);
   free(logfilename);
   SELF_PROFILE_END_FUNCTION;
}
