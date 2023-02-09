#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "self_profile.h"
#include "configuration_types.h"
#include "vftrace_state.h"

#include "signal_handling.h"
#include "filenames.h"
#include "configuration_print.h"
#include "logfile_header.h"
#include "logfile_prof_table.h"
#include "logfile_mpi_table.h"
#include "logfile_stacklist.h"
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

#define N_LOGFILE_TYPES 7

enum {LOG_MAIN, LOG_MINMAX, LOG_GROUPED,
      LOG_MPI, LOG_CUDA, LOG_ACCPROF, LOG_HWPROF};

typedef struct {
  FILE *fp[N_LOGFILE_TYPES];
} vftr_logfile_fp_t;

char *vftr_get_logfile_name(config_t config, char *type) {
   char *filename_base = vftr_create_filename_base(config, -1, 1);
   int filename_base_len = strlen(filename_base);

   int type_len = type != NULL ? strlen(type) + 1 : 0; // Add one for '_' character

   char *extension = ".log";
   int extension_len = strlen(extension);

   // construct logfile name
   int total_len = filename_base_len +
                   type_len +
                   extension_len +
                   1; // null terminator
   char *logfile_name = (char*) malloc(total_len*sizeof(char));
   strcpy(logfile_name, filename_base);
   if (type != NULL) {
      strcat (logfile_name, "_");
      strcat (logfile_name, type);
   }
   strcat(logfile_name, extension);

   free(filename_base);
   return logfile_name;
}

FILE *vftr_open_logfile(char *filename) {
   FILE *fp = fopen(filename, "w");
   if (fp == NULL) {
      perror(filename);
      vftr_abort(0);
   }
   return fp;
}

FILE *vftr_get_this_fp (char *type, FILE *fp_main) {
  if (vftrace.config.profile_table.separate.value) {
    char *this_logfile = vftr_get_logfile_name (vftrace.config, type);
    FILE *fp = vftr_open_logfile(this_logfile);
    free(this_logfile);
    return fp;
    //return vftr_open_logfile(this_logfile);
  } else {
    return fp_main;
  }
}

vftr_logfile_fp_t vftr_open_separate_logfiles (config_t config) {
  vftr_logfile_fp_t all_fp;
   
  char *logfilename_main = vftr_get_logfile_name(vftrace.config, NULL);
  all_fp.fp[LOG_MAIN] = vftr_open_logfile(logfilename_main);
  free(logfilename_main);
  FILE *fp_main = all_fp.fp[LOG_MAIN];

  all_fp.fp[LOG_MINMAX] = config.profile_table.show_minmax_summary.value ?
                          vftr_get_this_fp ("minmax", fp_main) : NULL;
  all_fp.fp[LOG_GROUPED] = config.name_grouped_profile_table.show_table.value ?
                           vftr_get_this_fp ("namegroup", fp_main) : NULL;
#ifdef _MPI
  int mpi_initialized;
  PMPI_Initialized(&mpi_initialized);
  all_fp.fp[LOG_MPI] = config.mpi.show_table.value && mpi_initialized ?
                       vftr_get_this_fp ("mpi", fp_main) : NULL;
#else
  all_fp.fp[LOG_MPI] = NULL;
#endif

#if defined(_CUDA)
  all_fp.fp[LOG_CUDA] = config.cuda.show_table.value ?
                        vftr_get_this_fp ("cuda", fp_main) : NULL;
#else
  all_fp.fp[LOG_CUDA] = NULL;
#endif

#if defined(_ACCPROF)
  all_fp.fp[LOG_ACCPROF] = config.accprof.show_table.value ?
                           vftr_get_this_fp ("accprof", fp_main) : NULL;
#else
  all_fp.fp[LOG_ACCPROF] = NULL;
#endif 

  all_fp.fp[LOG_HWPROF] = config.hwprof.active.value ?
                          vftr_get_this_fp ("hwprof", fp_main) : NULL;
  return all_fp;
}

void vftr_close_separate_logfiles (vftr_logfile_fp_t all_fp) {
  FILE *fp_main = all_fp.fp[LOG_MAIN];
  for (int i = 1; i < N_LOGFILE_TYPES; i++) {
    if (all_fp.fp[i] != NULL && all_fp.fp[i] != fp_main) fclose(all_fp.fp[i]);
  }
  fclose(fp_main);
}

void vftr_write_other_tables (vftrace_t vftrace, vftr_logfile_fp_t all_fp) {
   if (all_fp.fp[LOG_MINMAX] != NULL) vftr_write_minmax_summary (all_fp.fp[LOG_MINMAX], vftrace);
   if (all_fp.fp[LOG_GROUPED] != NULL) {
      collated_stacktree_t namegrouped_collated_stacktree =
         vftr_collated_stacktree_group_by_name(&vftrace.process.collated_stacktree);
      vftr_write_logfile_name_grouped_profile_table (all_fp.fp[LOG_GROUPED],
                                                    namegrouped_collated_stacktree,
                                                    vftrace.config);
      vftr_collated_stacktree_free(&namegrouped_collated_stacktree);
   }

   if (all_fp.fp[LOG_MPI] != NULL) {
      vftr_write_logfile_mpi_table (all_fp.fp[LOG_MPI],
                                    vftrace.process.collated_stacktree,
                                    vftrace.config);
    }

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

void vftr_write_warnings (vftrace_t vftrace, vftr_logfile_fp_t all_fp) {
#ifdef _MPI
   int mpi_initialized;
   PMPI_Initialized(&mpi_initialized);
   if (!mpi_initialized)
     fprintf (all_fp.fp[LOG_MPI], "The MPI interface is active, but MPI has not been initialized\n");
#endif   

#ifdef _CUDA
   if (vftrace.cuda_state.n_devices == 0)
      fprintf (all_fp.fp[LOG_CUDA], "The CUpti interface is enabled, but no GPU devices were found.\n");
#endif

#ifdef _ACCPROF
   if (vftrace.accprof_state.n_devices == 0) {
      fprintf (all_fp.fp[LOG_ACCPROf], "\nThe ACCProf interface is enabled, but no GPU devices were found.\n");
   } else if (!vftr_has_accprof_events (vftrace.process.collated_stacktree)) {
      fprintf (fp[LOG_ACCPROF], "\nNo OpenACC events have been registered.\n");
   } else if (vftrace.config.accprof.show_table.value) {
      if (vftrace.accprof_state.n_open_wait_queues > 0) {
         fprintf (fp[LOG_ACCPROF], "\nWarning: There are %d unresolved OpenACC wait regions.\n", 
                  vftrace.accprof_state.n_open_wait_queues);
      }
   }
#endif
}

void vftr_write_logfile_prologue (vftrace_t vftrace, vftr_logfile_fp_t all_fp, long long runtime) {
  for (int i = 0; i < N_LOGFILE_TYPES; i++) {
     if (all_fp.fp[i] != NULL) vftr_write_logfile_header (all_fp.fp[i], vftrace.timestrings);
  }

  if (vftrace.signal_received > 0) vftr_write_signal_message (all_fp.fp[LOG_MAIN]);
 
  vftr_write_logfile_summary (all_fp.fp[LOG_MAIN], vftrace.process, vftrace.size, runtime); 
  vftr_write_warnings (vftrace, all_fp);
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

   vftr_logfile_fp_t all_fp = vftr_open_separate_logfiles (vftrace.config);

   vftr_write_logfile_prologue (vftrace, all_fp, runtime);

   if (vftrace.config.profile_table.show_table.value) {
      vftr_write_logfile_profile_table(all_fp.fp[LOG_MAIN],
                                       vftrace.process.collated_stacktree,
                                       vftrace.config);
   }

   vftr_write_other_tables (vftrace, all_fp);

   vftr_write_logfile_global_stack_list(all_fp.fp[LOG_MAIN], vftrace.process.collated_stacktree);

   vftr_write_logfile_epilogue (vftrace, all_fp);
   vftr_close_separate_logfiles (all_fp);
   SELF_PROFILE_END_FUNCTION;
}
