#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "self_profile.h"
#include "configuration_types.h"
#include "vftrace_state.h"
#include "logfile_common_types.h"

#include "signal_handling.h"
#include "filenames.h"
#include "configuration_print.h"
#include "logfile_prof_table.h"
#include "logfile_mpi_table.h"
#include "logfile_stacklist.h"
#include "logfile_common.h"
#include "collate_stacks.h"
#include "search.h"
#include "range_expand.h"
#include "hwprof_logfile.h"
#include "minmax_summary.h"
#ifdef _CUDA
#include "cuda_logfile.h"
#endif
#ifdef _ACCPROF
#include "accprof_logfile.h"
#endif

void vftr_write_logfile_other_tables (vftrace_t vftrace, vftr_logfile_fp_t all_fp) {
#ifdef _MPI
   if (all_fp.fp[LOG_MINMAX] != NULL) vftr_write_minmax_summary (all_fp.fp[LOG_MINMAX], vftrace);
#endif
   if (all_fp.fp[LOG_GROUPED] != NULL) {
      collated_stacktree_t namegrouped_collated_stacktree =
         vftr_collated_stacktree_group_by_name(&vftrace.process.collated_stacktree);
      vftr_write_logfile_name_grouped_profile_table (all_fp.fp[LOG_GROUPED],
                                                    namegrouped_collated_stacktree,
                                                    vftrace.config);
      vftr_collated_stacktree_free(&namegrouped_collated_stacktree);
   }

#ifdef _MPI
   if (all_fp.fp[LOG_MPI] != NULL) {
      vftr_write_logfile_mpi_table (all_fp.fp[LOG_MPI],
                                    vftrace.process.collated_stacktree,
                                    vftrace.config);
    }
#endif

#ifdef _CUDA
   if (all_fp.fp[LOG_CUDA] != NULL) {
      vftr_write_logfile_cuda_table (all_fp.fp[LOG_CUDA],
                                     vftrace.process.collated_stacktree,
                                     vftrace.config);
   }
#endif

#ifdef _ACCPROF
   if (all_fp.fp[LOG_ACCPROF] != NULL) {
      vftr_write_logfile_accprof_grouped_table (all_fp.fp[LOG_ACCPROF],
                                                vftrace.process.collated_stacktree,
                                                vftrace.config);
      if (vftrace.config.accprof.show_event_details.value) {
         vftr_write_logfile_accprof_event_table (all_fp.fp[LOG_ACCPROF],
                                                 vftrace.process.collated_stacktree,
                                                 vftrace.config);
      }
   }
#endif

   if (all_fp.fp[LOG_HWPROF] != NULL) {
      if (vftrace.hwprof_state.n_observables > 0 && vftrace.config.hwprof.show_observables.value) {
      vftr_write_hwprof_observables_table (all_fp.fp[LOG_HWPROF],
                                           vftrace.process.collated_stacktree,
                                           vftrace.config); 
      }
      if (vftrace.hwprof_state.n_counters > 0 && vftrace.config.hwprof.show_counters.value) {
         vftr_write_logfile_hwprof_counter_table (all_fp.fp[LOG_HWPROF],
                                                  vftrace.process.collated_stacktree,
                                                  vftrace.config);
      }

      if (vftrace.config.hwprof.show_observables.value && vftrace.config.hwprof.show_summary.value) {
         vftr_write_hwprof_observables_logfile_summary (all_fp.fp[LOG_HWPROF],
                                                        vftrace.process.collated_stacktree);
         fprintf (all_fp.fp[LOG_HWPROF], "\n");
      }
      if (vftrace.config.hwprof.show_counters.value && vftrace.config.hwprof.show_summary.value) {
         vftr_write_hwprof_counter_logfile_summary (all_fp.fp[LOG_HWPROF],
                                                    vftrace.process.collated_stacktree);
         fprintf (all_fp.fp[LOG_HWPROF], "\n");
      }
   }
}

void vftr_write_logfile_epilogue (vftrace_t vftrace, vftr_logfile_fp_t all_fp) {

#ifdef _CUDA
   if (vftrace.config.cuda.show_table.value) {
      vftr_write_logfile_cbid_names (all_fp.fp[LOG_CUDA], vftrace.process.collated_stacktree);
   }
#endif

#ifdef _ACCPROF
   if (vftrace.config.accprof.show_event_details.value) vftr_write_logfile_accev_names (all_fp.fp[LOG_ACCPROF]);
#endif


   if (vftrace.config.print_config.value) {
      vftr_print_config(all_fp.fp[LOG_MAIN], vftrace.config, true);
   }
}

void vftr_write_logfile(vftrace_t vftrace, long long runtime) {
   SELF_PROFILE_START_FUNCTION;
   // only process 0 writes the summary logfile
   if (vftrace.process.processID != 0) {
      SELF_PROFILE_END_FUNCTION;
      return;
   }

   vftr_logfile_fp_t all_fp = vftr_logfile_open_fps (vftrace.config, -1, 0);

   vftr_write_logfile_prologue (true, vftrace, all_fp, runtime);

   if (vftrace.config.profile_table.show_table.value) {
      vftr_write_logfile_profile_table(all_fp.fp[LOG_MAIN],
                                       vftrace.process.collated_stacktree,
                                       vftrace.config);
   }

   vftr_write_logfile_other_tables (vftrace, all_fp);

   vftr_write_logfile_global_stack_list(all_fp.fp[LOG_MAIN], vftrace.process.collated_stacktree);

   vftr_write_logfile_epilogue (vftrace, all_fp);
   vftr_logfile_close_fp (all_fp);
   SELF_PROFILE_END_FUNCTION;
}
