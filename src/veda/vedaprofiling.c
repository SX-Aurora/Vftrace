#include "vedaprofiling_types.h"

vedaprofile_t vftr_new_vedaprofiling() {
   vedaprofile_t prof;
   prof.n_calls = 0;
   prof.HtoD_bytes = 0ll;
   prof.DtoH_bytes = 0ll;
   prof.H_bytes = 0ll;
   prof.acc_HtoD_bw = 0.0;
   prof.acc_DtoH_bw = 0.0;
   prof.acc_H_bw = 0.0;
   prof.total_time_nsec = 0ll;
   prof.overhead_nsec = 0ll;

   return prof;
}

vedaprofile_t vftr_add_vedaprofiles(vedaprofile_t profA, vedaprofile_t profB) {
   vedaprofile_t profC;
   profC.n_calls = profA.n_calls + profB.n_calls;
   profC.HtoD_bytes = profA.HtoD_bytes + profB.HtoD_bytes;
   profC.DtoH_bytes = profA.DtoH_bytes + profB.DtoH_bytes;
   profC.H_bytes = profA.H_bytes + profB.H_bytes;
   profC.acc_HtoD_bw = profA.acc_HtoD_bw + profB.acc_HtoD_bw;
   profC.acc_DtoH_bw = profA.acc_DtoH_bw + profB.acc_DtoH_bw;
   profC.acc_H_bw = profA.acc_H_bw + profB.acc_H_bw;
   profC.total_time_nsec = profA.total_time_nsec + profB.total_time_nsec;
   profC.overhead_nsec = profA.overhead_nsec + profB.overhead_nsec;

   return profC;
}

void vftr_accumulate_vedaprofiling_overhead(vedaprofile_t *prof,
                                            long long overhead_nsec) {
   prof->overhead_nsec += overhead_nsec;
}

void vftr_vedaprofiling_free(vedaprofile_t *prof_ptr) {
   (void) prof_ptr;
}

long long *vftr_get_total_veda_overhead(stacktree_t stacktree, int nthreads) {
   // accumulate the veda overhead for each thread separately
   long long *overheads_nsec = (long long*) malloc(nthreads*sizeof(long long));
   for (int ithread=0; ithread<nthreads; ithread++) {
      overheads_nsec[ithread] = 0ll;
   }

   int nstacks = stacktree.nstacks;
   for (int istack=0; istack<nstacks; istack++) {
      stack_t *stack = stacktree.stacks+istack;
      int nprofs = stack->profiling.nprofiles;
      for (int iprof=0; iprof<nprofs; iprof++) {
         profile_t *prof = stack->profiling.profiles+iprof;
         int threadID = prof->threadID;
         overheads_nsec[threadID] += prof->vedaprof.overhead_nsec;
      }
   }
   return overheads_nsec;
}

long long vftr_get_total_collated_veda_overhead(collated_stacktree_t stacktree) {
   long long overheads_nsec = 0ll;

   int nstacks = stacktree.nstacks;
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack = stacktree.stacks+istack;
      collated_profile_t *prof = &(stack->profile);
      overheads_nsec += prof->vedaprof.overhead_nsec;
   }
   return overheads_nsec;
}

void vftr_print_vedaprofiling(FILE *fp, vedaprofile_t vedaprof) {
   fprintf(fp, "ncalls: %d, bytes: %lld/%lld/%lld, time: %.6lf\n",
           vedaprof.ncalls,
           vedaprof.HtoD_bytes, vedaprof.DtoH_bytes, vedaprof.H_bytes,
           vedaprof.total_time_nsec*1.0e-9);
}
