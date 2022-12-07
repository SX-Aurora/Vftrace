#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "self_profile.h"
#include "configuration_types.h"
#include "vftrace_state.h"

#include "filenames.h"
#include "ranklogfile_header.h"
#include "ranklogfile_prof_table.h"
#ifdef _MPI
#include "ranklogfile_mpi_table.h"
#endif
#include "logfile_stacklist.h"
#include "search.h"
#include "configuration_print.h"
#include "range_expand.h"
#ifdef _CUDA
#include "cuda_ranklogfile.h"
#endif
#ifdef _ACCPROF
#include "accprof_ranklogfile.h"
#endif
#ifdef _VEDA
#include "ranklogfile_veda_table.h"
#endif


static bool vftr_rank_needs_ranklogfile(config_t config, int rank) {
   char *rangestr = config.logfile_for_ranks.value;
   if (!strcmp(rangestr, "all")) {
      return true;
   }
   if (!strcmp(rangestr, "none")) {
      return false;
   }
   int nranks = 0;
   int *ranklist = vftr_expand_rangelist(rangestr, &nranks);
   int idx = vftr_binary_search_int(nranks, ranklist, rank);
   free(ranklist);
   if (idx == -1) {
      return false;
   } else {
      return true;
   }
}

char *vftr_get_ranklogfile_name(config_t config, int rankID, int nranks) {
   char *filename_base = vftr_create_filename_base(config, rankID, nranks);
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

FILE *vftr_open_ranklogfile(char *filename) {
   FILE *fp = fopen(filename, "w");
   if (fp == NULL) {
      perror(filename);
      abort();
   }
   return fp;
}

void vftr_write_ranklogfile(vftrace_t vftrace, long long runtime) {
   SELF_PROFILE_START_FUNCTION;
   if (!vftr_rank_needs_ranklogfile(vftrace.config, vftrace.process.processID)) {
      SELF_PROFILE_END_FUNCTION;
      return;
   }

   char *logfilename = vftr_get_ranklogfile_name(vftrace.config,
                                                 vftrace.process.processID,
                                                 vftrace.process.nprocesses);
   FILE *fp = vftr_open_ranklogfile(logfilename);

   vftr_write_ranklogfile_header(fp, vftrace.timestrings);

   vftr_write_ranklogfile_summary(fp, vftrace.process,
                                  vftrace.size, runtime);

   if (vftrace.config.profile_table.show_table.value) {
      vftr_write_ranklogfile_profile_table(fp, vftrace.process.stacktree,
                                           vftrace.config);
   }

#ifdef _MPI
   int mpi_initialized;
   PMPI_Initialized(&mpi_initialized);
   if (vftrace.config.mpi.show_table.value && mpi_initialized) {
      vftr_write_ranklogfile_mpi_table(fp, vftrace.process.stacktree,
                                       vftrace.config);
   }
#endif

#ifdef _CUDA
   if (vftrace.cuda_state.n_devices == 0) {
      fprintf (fp, "The CUpti interface is enabled, but no GPU devices were found.\n");
   } else if (vftrace.config.cuda.show_table.value) {
      vftr_write_ranklogfile_cuda_table(fp, vftrace.process.stacktree, vftrace.config);
   }
#endif

#ifdef _ACCPROF
   if (vftrace.accprof_state.n_devices == 0) {
      fprintf (fp, "The ACCProf interface is enabled, but no GPU devices were found.\n");
   } else if (vftrace.config.accprof.show_table.value) {
      vftr_write_ranklogfile_accprof_grouped_table (fp, vftrace.process.stacktree, vftrace.config);
      if (vftrace.config.accprof.show_event_details.value) {
         vftr_write_ranklogfile_accprof_event_table (fp, vftrace.process.stacktree, vftrace.config);
      }
   }
#endif

#ifdef _VEDA
   // TODO CONFIG FOR VEDA
   vftr_write_ranklogfile_veda_table(fp, vftrace.process.stacktree, vftrace.config);
#endif

   vftr_write_logfile_global_stack_list(fp, vftrace.process.collated_stacktree);

   // print config info
   if (vftrace.config.print_config.value) {
      vftr_print_config(fp, vftrace.config, true);
   }

#ifdef _CUDA
   if (vftrace.config.cuda.show_table.value) {
      vftr_write_ranklogfile_cbid_names (fp, vftrace.process.stacktree);
   }
#endif

   fclose(fp);
   free(logfilename);
   SELF_PROFILE_END_FUNCTION;
}
