#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "self_profile.h"
#include "environment_types.h"
#include "vftrace_state.h"

#include "filenames.h"
#include "environment.h"
#include "ranklogfile_header.h"
#include "ranklogfile_prof_table.h"
#include "ranklogfile_mpi_table.h"
#include "logfile_stacklist.h"
#include "search.h"
#include "range_expand.h"
#ifdef _CUPTI
#include "gpu_info.h"
#endif


static bool vftr_rank_needs_ranklogfile(environment_t environment, int rank) {
   char *rangestr = environment.logfile_for_ranks.value.string_val;
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

char *vftr_get_ranklogfile_name(environment_t environment, int rankID, int nranks) {
   char *filename_base = vftr_create_filename_base(environment, rankID, nranks);
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
   if (!vftr_rank_needs_ranklogfile(vftrace.environment, vftrace.process.processID)) {
      SELF_PROFILE_END_FUNCTION;
      return;
   }

   char *logfilename = vftr_get_ranklogfile_name(vftrace.environment,
                                                 vftrace.process.processID,
                                                 vftrace.process.nprocesses);
   FILE *fp = vftr_open_ranklogfile(logfilename);

   vftr_write_ranklogfile_header(fp, vftrace.timestrings);

   vftr_write_ranklogfile_summary(fp, vftrace.process,
                                  vftrace.size, runtime);

#ifdef _CUPTI
   vftr_write_gpu_info (fp, vftrace.cupti_state.n_devices);
#endif

   vftr_write_ranklogfile_profile_table(fp, vftrace.process.stacktree,
                                        vftrace.environment);

#ifdef _MPI
   vftr_write_ranklogfile_mpi_table(fp, vftrace.process.stacktree,
                                    vftrace.environment);
#endif

   vftr_write_logfile_global_stack_list(fp, vftrace.process.collated_stacktree);

   // print environment info
   vftr_print_environment(fp, vftrace.environment);

   fclose(fp);
   free(logfilename);
   SELF_PROFILE_END_FUNCTION;
}
