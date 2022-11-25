#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "self_profile.h"
#include "size_types.h"
#include "timer_types.h"
#include "table_types.h"
#include "vftrace_state.h"

#include "filenames.h"
#include "license.h"
#include "config.h"
#include "stacks.h"
#include "collate_stacks.h"
#include "profiling_types.h"
#include "profiling.h"
#include "collated_callprofiling.h"
#include "mpiprofiling.h"
#include "ompprofiling.h"
#include "tables.h"
#include "misc_utils.h"

#ifdef _CUPTI
#include "cupti_logfile.h"
#include "cuptiprofiling.h"
#endif

#ifdef _VEDA
#include "vedaprofiling.h"
#endif

void vftr_write_logfile_header(FILE *fp, time_strings_t timestrings) {
   SELF_PROFILE_START_FUNCTION;
   fprintf(fp, "%s\n", PACKAGE_STRING);
   fprintf(fp, "Runtime profile for application:\n");
   fprintf(fp, "Start Date: %s\n", timestrings.start_time);
   fprintf(fp, "End Date:   %s\n\n", timestrings.end_time);
   vftr_print_licence(fp);
   SELF_PROFILE_END_FUNCTION;
}

void vftr_write_logfile_summary(FILE *fp, process_t process,
                                    vftr_size_t vftrace_size,
                                    long long runtime) {
   SELF_PROFILE_START_FUNCTION;
   double runtime_sec = runtime * 1.0e-9;

   // The application runtime is the runtime minus the
   long long total_master_overhead = 0ll;
   long long call_overhead =
      vftr_get_total_collated_call_overhead(process.collated_stacktree);
#ifdef _MPI
   long long mpi_overhead =
      vftr_get_total_collated_mpi_overhead(process.collated_stacktree);
#endif
#ifdef _OMP
   long long omp_overhead =
      vftr_get_total_collated_omp_overhead(process.collated_stacktree);
#endif
#ifdef _CUPTI
   long long cupti_overhead = 
      vftr_get_total_collated_cupti_overhead(process.collated_stacktree);
#endif
#ifdef _VEDA
   long long veda_overhead = 
      vftr_get_total_collated_veda_overhead(process.collated_stacktree);
#endif

      total_master_overhead += call_overhead;
#ifdef _MPI
      total_master_overhead += mpi_overhead;
#endif
#ifdef _OMP
      total_master_overhead += omp_overhead;
#endif
#ifdef _CUPTI
      total_master_overhead += cupti_overhead;
#endif
#ifdef _VEDA
      total_master_overhead += veda_overhead;
#endif
   
   double total_master_overhead_sec = total_master_overhead * 1.0e-9;
   double apptime_sec = runtime_sec - total_master_overhead_sec / process.nprocesses;

   fprintf(fp, "\n");
#ifdef _MPI
   fprintf(fp, "Nr. of MPI ranks:     %8d\n", process.nprocesses);
#endif
   fprintf(fp, "Total runtime:        %8.2lf s\n", runtime_sec);
   fprintf(fp, "Application time:     %8.2lf s\n", apptime_sec);
   fprintf(fp, "Overhead:             %8.2lf s\n",
           total_master_overhead_sec/process.nprocesses);
   fprintf(fp, "   Function hooks:    %8.2lf s\n",
           call_overhead*1.0e-9/process.nprocesses);
#ifdef _MPI
   fprintf(fp, "   MPI wrappers:      %8.2lf s\n",
           mpi_overhead*1.0e-9/process.nprocesses);
#endif
#ifdef _OMP
   fprintf(fp, "   OMP callbacks:     %8.2lf s\n",
           omp_overhead*1.0e-9/process.nprocesses);
#endif
#ifdef _CUPTI
   fprintf(fp, "   CUPTI callbacks:   %8.2lf s\n",
           cupti_overhead * 1e-9 / process.nprocesses);
#endif
#ifdef _VEDA
   fprintf(fp, "   VEDA callbacks:    %8.2lf s\n",
           veda_overhead*1.0e-9/process.nprocesses);
#endif

#ifdef _CUPTI
   float total_compute_sec, total_memcpy_sec, total_other_sec;
   vftr_get_total_cupti_times_for_logfile (process.collated_stacktree,
                                           &total_compute_sec, &total_memcpy_sec, &total_other_sec);
   fprintf (fp, "Total CUDA time:      %8.2f s\n", total_compute_sec + total_memcpy_sec + total_other_sec);
   fprintf (fp, "   Compute:           %8.2f s\n", total_compute_sec);
   fprintf (fp, "   Memcpy:            %8.2f s\n", total_memcpy_sec);
   fprintf (fp, "   Other:             %8.2f s\n", total_other_sec);
#endif

#ifdef _VEDA
   // TODO: VEDA header
#endif
   char *unit = vftr_byte_unit(vftrace_size.total);
   double vftrace_size_double = (double) vftrace_size.total;
   while (vftrace_size_double > 1024.0) {vftrace_size_double /= 1024;}
   fprintf(fp, "Vftrace used memory:   %7.2lf %s\n", vftrace_size_double, unit);
   free(unit);
   SELF_PROFILE_END_FUNCTION;
}
