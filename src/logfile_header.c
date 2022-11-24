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

#ifdef _CUDA
#include "cuda_logfile.h"
#include "cudaprofiling.h"
#endif

#ifdef _ACCPROF
#include "accprof_logfile.h"
#include "accprofiling.h"
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
#ifdef _CUDA
   long long cuda_overhead =
      vftr_get_total_collated_cuda_overhead(process.collated_stacktree);
#endif
#ifdef _ACCPROF
   long long accprof_overhead =
      vftr_get_total_collated_accprof_overhead(process.collated_stacktree);
#endif

      total_master_overhead += call_overhead;
#ifdef _MPI
      total_master_overhead += mpi_overhead;
#endif
#ifdef _OMP
      total_master_overhead += omp_overhead;
#endif
#ifdef _CUDA
      total_master_overhead += cuda_overhead;
#endif
#ifdef _ACCPROF
      total_master_overhead += accprof_overhead;
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
#ifdef _CUDA
   fprintf (fp, "   CUDA callbacks:  %8.2lf s\n",
            cuda_overhead * 1e-9 / process.nprocesses);
#endif
#ifdef _ACCPROF
   fprintf (fp, "   OpenACC callbacks:  %8.2lf s\n",
            accprof_overhead * 1e-9 / process.nprocesses);
#endif

#ifdef _CUDA
   float total_compute_sec_cuda, total_memcpy_sec_cuda, total_other_sec_cuda;
   vftr_get_total_cuda_times_for_logfile (process.collated_stacktree,
                                           &total_compute_sec_cuda,
                                           &total_memcpy_sec_cuda,
                                           &total_other_sec_cuda);
   fprintf (fp, "Total CUDA time:      %8.2f s\n",
                total_compute_sec_cuda + total_memcpy_sec_cuda + total_other_sec_cuda);
   fprintf (fp, "   Compute:           %8.2f s\n", total_compute_sec_cuda);
   fprintf (fp, "   Memcpy:            %8.2f s\n", total_memcpy_sec_cuda);
   fprintf (fp, "   Other:             %8.2f s\n", total_other_sec_cuda);
#endif

#ifdef _ACCPROF
   double total_compute_sec_accprof, total_memcpy_sec_accprof, total_other_sec_accprof;
   vftr_get_total_accprof_times_for_logfile (process.collated_stacktree,
					     &total_compute_sec_accprof,
					     &total_memcpy_sec_accprof,
					     &total_other_sec_accprof);
   fprintf (fp, "Total OpenACC time:   %8.2f s\n",
                total_compute_sec_accprof + total_memcpy_sec_accprof + total_other_sec_accprof);
   fprintf (fp, "  Compute:            %8.2f s\n", total_compute_sec_accprof);
   fprintf (fp, "  Memcpy:             %8.2f s\n", total_memcpy_sec_accprof);
   fprintf (fp, "  Other:              %8.2f s\n", total_other_sec_accprof);

#endif

   char *unit = vftr_byte_unit(vftrace_size.total);
   double vftrace_size_double = (double) vftrace_size.total;
   while (vftrace_size_double > 1024.0) {vftrace_size_double /= 1024;}
   fprintf(fp, "Vftrace used memory:   %7.2lf %s\n", vftrace_size_double, unit);
   free(unit);
   SELF_PROFILE_END_FUNCTION;
}
