#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "self_profile.h"
#include "configuration_types.h"
#include "vftrace_state.h"

#include "signal_handling.h"
#include "filenames.h"
#include "logfile_common.h"
#include "logfile_stacklist.h"
#include "ranklogfile_header.h"
#include "ranklogfile_prof_table.h"
#include "ranklogfile_mpi_table.h"
#include "search.h"
#include "configuration_print.h"
#include "range_expand.h"
#include "hwprof_ranklogfile.h"
#ifdef _CUDA
#include "cuda_ranklogfile.h"
#endif
#ifdef _ACCPROF
#include "accprof_ranklogfile.h"
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

FILE *vftr_open_ranklogfile(char *filename) {
   FILE *fp = fopen(filename, "w");
   if (fp == NULL) {
      perror(filename);
      vftr_abort(0);
   }
   return fp;
}

void vftr_write_ranklogfile_other_tables (vftrace_t vftrace, vftr_logfile_fp_t all_fp) {

// Min/Max summaries and grouped tables are irrelevant for ranklogfiles.

#ifdef _MPI
   if (all_fp.fp[LOG_MPI] != NULL) {
         vftr_write_ranklogfile_mpi_table(all_fp.fp[LOG_MPI], vftrace.process.stacktree,
                                          vftrace.config);
   }
#endif

#ifdef _CUDA
   if (all_fp.fp[LOG_CUDA] != NULL) {
      vftr_write_ranklogfile_cuda_table(all_fp.fp[LOG_CUDA], vftrace.process.stacktree, vftrace.config);
   }
#endif

#ifdef _ACCPROF
   if (all_fp.fp[LOG_ACCPROF] != NULL) {
      vftr_write_ranklogfile_accprof_grouped_table (all_fp.fp[LOG_ACCPROF], vftrace.process.stacktree, vftrace.config);
      if (vftrace.config.accprof.show_event_details.value) {
         vftr_write_ranklogfile_accprof_event_table (all_fp.fp[LOG_ACCPROF], vftrace.process.stacktree, vftrace.config);
      }
   }
#endif

   if (all_fp.fp[LOG_HWPROF] != NULL) {
      if (vftrace.hwprof_state.n_observables > 0 && vftrace.config.hwprof.show_observables.value) {
         vftr_write_ranklogfile_hwprof_obs_table (all_fp.fp[LOG_HWPROF], vftrace.process.stacktree, vftrace.config);
         fprintf (all_fp.fp[LOG_HWPROF], "\n");
      }
      if (vftrace.hwprof_state.n_counters > 0 && vftrace.config.hwprof.show_counters.value) {
         vftr_write_ranklogfile_hwprof_counter_table (all_fp.fp[LOG_HWPROF], vftrace.process.stacktree, vftrace.config);
         fprintf (all_fp.fp[LOG_HWPROF], "\n");
      }

      if (vftrace.config.hwprof.show_observables.value && vftrace.config.hwprof.show_summary.value) {
         vftr_write_hwprof_observables_ranklogfile_summary (all_fp.fp[LOG_HWPROF], vftrace.process.stacktree);
         fprintf (all_fp.fp[LOG_HWPROF], "\n");
      }
      if (vftrace.config.hwprof.show_counters.value && vftrace.config.hwprof.show_summary.value) {
         vftr_write_hwprof_counter_ranklogfile_summary (all_fp.fp[LOG_HWPROF], vftrace.process.stacktree);
         fprintf (all_fp.fp[LOG_HWPROF], "\n");
      }
   }
}

void vftr_write_ranklogfile_epilogue (vftrace_t vftrace, vftr_logfile_fp_t all_fp) {
#ifdef _CUDA
   if (vftrace.config.cuda.show_table.value) {
      vftr_write_ranklogfile_cbid_names (all_fp.fp[LOG_CUDA], vftrace.process.stacktree);
   }
#endif

   if (vftrace.config.print_config.value) {
      vftr_print_config(all_fp.fp[LOG_MAIN], vftrace.config, true);
   }
}

void vftr_write_ranklogfile(vftrace_t vftrace, long long runtime) {
   SELF_PROFILE_START_FUNCTION;
   if (!vftr_rank_needs_ranklogfile(vftrace.config, vftrace.process.processID)) {
      SELF_PROFILE_END_FUNCTION;
      return;
   }

   vftr_logfile_fp_t all_fp = vftr_logfile_open_fps (vftrace.config,
                                                     vftrace.process.processID,
                                                     vftrace.process.nprocesses);

   vftr_write_logfile_prologue (false, vftrace, all_fp, runtime);

   if (vftrace.config.profile_table.show_table.value) {
      vftr_write_ranklogfile_profile_table(all_fp.fp[LOG_MAIN], vftrace.process.stacktree,
                                           vftrace.config);
   }

   vftr_write_ranklogfile_other_tables (vftrace, all_fp);

   vftr_write_logfile_global_stack_list(all_fp.fp[LOG_MAIN], vftrace.process.collated_stacktree);

   vftr_write_ranklogfile_epilogue (vftrace, all_fp);
   vftr_logfile_close_fp (all_fp);
   SELF_PROFILE_END_FUNCTION;
}
