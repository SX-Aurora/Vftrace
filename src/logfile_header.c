#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "timer_types.h"
#include "table_types.h"
#include "environment_types.h"
#include "vftrace_state.h"

#include "filenames.h"
#include "license.h"
#include "config.h"
#include "log_profile.h"
#include "environment.h"
#include "stacks.h"
#include "collate_stacks.h"
#include "tables.h"
#include "overheadprofiling_types.h"
#include "overheadprofiling.h"

void vftr_write_logfile_header(FILE *fp, time_strings_t timestrings,
                               environment_t environment) {
   fprintf(fp, "%s\n", PACKAGE_STRING);
   fprintf(fp, "Runtime profile for application:\n");
   fprintf(fp, "Start Date: %s\n", timestrings.start_time);
   fprintf(fp, "End Date:   %s\n\n", timestrings.end_time);
   // print the full license if requested by the environment
   if (environment.license_verbose.value.bool_val) {
      vftr_print_licence(fp);
   } else {
      vftr_print_licence_short(fp, environment.license_verbose.name);
   }
}

void vftr_write_logfile_summary(FILE *fp, process_t process, long long runtime) {
   double runtime_sec = runtime * 1.0e-6;

   // get the different accumulated overheads
   // The application runtime is the runtime minus the
   // sum of all overheads on the master thread
   long long total_master_overhead = 0ll;
   int nthreads = process.threadtree.nthreads;
   long long *hook_overheads = vftr_get_total_hook_overhead(process.stacktree, nthreads);
#ifdef _MPI
   long long *mpi_overheads = vftr_get_total_mpi_overhead(process.stacktree, nthreads);
#endif
#ifdef _OMP
   long long *omp_overheads = vftr_get_total_omp_overhead(process.stacktree, nthreads);
#endif
   for (int ithread=0; ithread<nthreads; ithread++) {
      if (process.threadtree.threads[ithread].master) {
         total_master_overhead += hook_overheads[ithread];
#ifdef _MPI
         total_master_overhead += mpi_overheads[ithread];
#endif
#ifdef _OMP
         total_master_overhead += omp_overheads[ithread];
#endif
      }
   }
   double total_master_overhead_sec = total_master_overhead*1.0e-6;
   double apptime_sec = runtime_sec - total_master_overhead_sec;

   fprintf(fp, "\n");
#ifdef _MPI
   fprintf(fp, "Nr. of MPI ranks:     %8d\n", process.nprocesses);
#endif
#ifdef _OMP
   fprintf(fp, "Nr. of OMP threads:   %8d\n", nthreads);
#endif
   fprintf(fp, "Total runtime:        %8.2lf s\n", runtime_sec);
   fprintf(fp, "Application time:     %8.2lf s\n", apptime_sec);
   fprintf(fp, "Overhead:             %8.2lf s\n", total_master_overhead_sec);
   if (nthreads == 1) {
      fprintf(fp, "   Function hooks:    %8.2lf s\n", hook_overheads[0]*1.0e-6);
#ifdef _MPI
      fprintf(fp, "   MPI wrappers:      %8.2lf s\n", mpi_overheads[0]*1.0e-6);
#endif
#ifdef _OMP
      fprintf(fp, "   OMP callbacks:     %8.2lf s\n", omp_overheads[0]*1.0e-6);
#endif
   } else {
      fprintf(fp, "   Function hooks:\n");
      for (int ithread=0; ithread<nthreads; ithread++) {
         fprintf(fp, "      Thread %d:      %8.2lf s\n",
                 ithread, hook_overheads[ithread]*1.0e-6);
      }
#ifdef _MPI
      fprintf(fp, "   MPI wrappers:\n");
      for (int ithread=0; ithread<nthreads; ithread++) {
         fprintf(fp, "      Thread %d:      %8.2lf s\n",
                 ithread, mpi_overheads[ithread]*1.0e-6);
      }
#endif
#ifdef _OMP
      fprintf(fp, "   OMP callbacks:\n");
      for (int ithread=0; ithread<nthreads; ithread++) {
         fprintf(fp, "      Thread %d:      %8.2lf s\n",
                 ithread, omp_overheads[ithread]*1.0e-6);
      }
#endif
   }
}
