#include <string.h>

#include "logfile_common_types.h"

#include "filenames.h"
#include "logfile.h"
#include "logfile_header.h"
#include "ranklogfile_header.h"
#include "signal_handling.h"

char *vftr_get_logfile_name(config_t config, char *type, int rankID, int nranks) {
   char *filename_base = vftr_create_filename_base(config, rankID, nranks);
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

FILE *vftr_get_this_logfile_fp (char *type, FILE *fp_main, int rankID, int nranks) {
  if (vftrace.config.profile_table.separate.value) {
    char *this_logfile;
    this_logfile = vftr_get_logfile_name (vftrace.config, type, rankID, nranks);
    FILE *fp = vftr_open_logfile(this_logfile);
    free(this_logfile);
    return fp;
  } else {
    return fp_main;
  }
}

vftr_logfile_fp_t vftr_logfile_open_fps (config_t config, int rankID, int nranks) {
  vftr_logfile_fp_t all_fp;
   
  char *logfilename_main = vftr_get_logfile_name(vftrace.config, NULL, rankID, nranks);
  all_fp.fp[LOG_MAIN] = vftr_open_logfile(logfilename_main);
  free(logfilename_main);
  FILE *fp_main = all_fp.fp[LOG_MAIN];

  if (rankID < 0) { // rankID >= 0 are ranklogfiles. Min/Max and grouped tables are only printed for the master rank.
     all_fp.fp[LOG_MINMAX] = config.profile_table.show_minmax_summary.value ?
                             vftr_get_this_logfile_fp ("minmax", fp_main, rankID, nranks) : NULL;
     all_fp.fp[LOG_GROUPED] = config.name_grouped_profile_table.show_table.value ?
                              vftr_get_this_logfile_fp ("namegroup", fp_main, rankID, nranks) : NULL;
  } else {
     all_fp.fp[LOG_MINMAX] = NULL;
     all_fp.fp[LOG_GROUPED] = NULL;
  }

#ifdef _MPI
  int mpi_initialized;
  PMPI_Initialized(&mpi_initialized);
  all_fp.fp[LOG_MPI] = config.mpi.show_table.value && mpi_initialized ?
                       vftr_get_this_logfile_fp ("mpi", fp_main, rankID, nranks) : NULL;
#else
  all_fp.fp[LOG_MPI] = NULL;
#endif

#if defined(_CUDA)
  all_fp.fp[LOG_CUDA] = config.cuda.show_table.value ?
                        vftr_get_this_logfile_fp ("cuda", fp_main, rankID, nranks) : NULL;
#else
  all_fp.fp[LOG_CUDA] = NULL;
#endif

#if defined(_ACCPROF)
  all_fp.fp[LOG_ACCPROF] = config.accprof.show_table.value ?
                           vftr_get_this_logfile_fp ("accprof", fp_main, rankID, nranks) : NULL;
#else
  all_fp.fp[LOG_ACCPROF] = NULL;
#endif 

  bool show_hwprof = (vftrace.hwprof_state.n_counters > 0 && vftrace.config.hwprof.show_counters.value) ||
                     (vftrace.hwprof_state.n_observables > 0 && vftrace.config.hwprof.show_observables.value);
  all_fp.fp[LOG_HWPROF] = show_hwprof ? vftr_get_this_logfile_fp ("hwprof", fp_main, rankID, nranks) : NULL;
  return all_fp;
}

void vftr_logfile_close_fp (vftr_logfile_fp_t all_fp) {
  FILE *fp_main = all_fp.fp[LOG_MAIN];
  for (int i = 1; i < N_LOGFILE_TYPES; i++) {
    if (all_fp.fp[i] != NULL && all_fp.fp[i] != fp_main) fclose(all_fp.fp[i]);
  }
  fclose(fp_main);
}

void vftr_write_logfile_warnings (vftrace_t vftrace, vftr_logfile_fp_t all_fp) {
#ifdef _MPI
   int mpi_initialized;
   PMPI_Initialized(&mpi_initialized);
   if (!mpi_initialized && all_fp.fp[LOG_MPI] != NULL)
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

void vftr_write_logfile_prologue (bool is_master_logfile, vftrace_t vftrace,
                                  vftr_logfile_fp_t all_fp, long long runtime) {
  for (int i = 0; i < N_LOGFILE_TYPES; i++) {
     if (all_fp.fp[i] != NULL) {
        vftr_write_logfile_header (all_fp.fp[i], vftrace.timestrings);
     }
  }

  if (vftrace.signal_received > 0) vftr_write_signal_message (all_fp.fp[LOG_MAIN]);
 
  if (is_master_logfile) {
     vftr_write_logfile_summary (all_fp.fp[LOG_MAIN], vftrace.process, vftrace.size, runtime); 
  } else {
     vftr_write_ranklogfile_summary (all_fp.fp[LOG_MAIN], vftrace.process, runtime);
  }
  vftr_write_logfile_warnings (vftrace, all_fp);
}


