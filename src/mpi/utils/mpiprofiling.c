#include <stdlib.h>
#include <stdbool.h>

#include <string.h>

#include "self_profile.h"
#include "vftrace_state.h"
#include "mpi_state_types.h"
#include "process_types.h"
#include "configuration_types.h"
#include "mpi_util_types.h"
#include "mpiprofiling_types.h"

#include "range_expand.h"
#include "search.h"

mpiprofile_t vftr_new_mpiprofiling() {
   mpiprofile_t prof;
   prof.nsendmessages = 0ll;
   prof.nrecvmessages = 0ll;
   prof.send_bytes = 0ll;
   prof.recv_bytes = 0ll;
   prof.acc_send_bw = 0.0;
   prof.acc_recv_bw = 0.0;
   prof.total_time_nsec = 0ll;
   prof.overhead_nsec = 0ll;

   return prof;
}

bool vftr_should_log_message_info(mpi_state_t mpi_state, int rank) {
   bool should = mpi_state.my_rank_in_prof;
   should = should && vftr_binary_search_int(mpi_state.nprof_ranks,
                                             mpi_state.prof_ranks,
                                             rank) >= 0;
   return should;
}

void vftr_accumulate_message_info(mpiprofile_t *prof_ptr,
                                  message_direction dir,
                                  long long count,
                                  int type_idx, int type_size,
                                  int rank, int tag,
                                  long long tstart,
                                  long long tend) {
   SELF_PROFILE_START_FUNCTION;
   (void) type_idx;
   (void) tag;
   int nbytes = count * type_size;
   long long time = tend - tstart;
   double bw = nbytes * 1.0e9 / time;

   if (dir == send) {
      prof_ptr->nsendmessages++;
      prof_ptr->send_bytes += nbytes;
      prof_ptr->acc_send_bw += bw;
   } else {
      prof_ptr->nrecvmessages++;
      prof_ptr->recv_bytes += nbytes;
      prof_ptr->acc_recv_bw += bw;
   }
   prof_ptr->total_time_nsec += time;
   SELF_PROFILE_END_FUNCTION;
}

mpiprofile_t vftr_add_mpiprofiles(mpiprofile_t profA, mpiprofile_t profB) {
   mpiprofile_t profC;
   profC.nsendmessages = profA.nsendmessages + profB.nsendmessages;
   profC.nrecvmessages = profA.nrecvmessages + profB.nrecvmessages;
   profC.send_bytes = profA.send_bytes + profB.send_bytes;
   profC.recv_bytes = profA.recv_bytes + profB.recv_bytes;
   profC.acc_send_bw= profA.acc_send_bw + profB.acc_send_bw;
   profC.acc_recv_bw = profA.acc_recv_bw + profB.acc_recv_bw;
   profC.total_time_nsec = profA.total_time_nsec + profB.total_time_nsec;
   profC.overhead_nsec = profA.overhead_nsec + profB.overhead_nsec;
   return profC;
}

void vftr_accumulate_mpiprofiling_overhead(mpiprofile_t *prof,
                                           long long overhead_nsec) {
   prof->overhead_nsec += overhead_nsec;
}

void vftr_mpiprofiling_free(mpiprofile_t *prof_ptr) {
   (void) prof_ptr;
}

void vftr_create_profiled_ranks_list(config_t config,
                                     process_t process,
                                     mpi_state_t *mpi_state) {
   SELF_PROFILE_START_FUNCTION;
   char *rangestr = config.mpi.only_for_ranks.value;
   if (!strcmp(rangestr, "all")) {
      mpi_state->nprof_ranks = process.nprocesses;
      mpi_state->prof_ranks = (int*) malloc(mpi_state->nprof_ranks*sizeof(int));
      for (int irank=0; irank<mpi_state->nprof_ranks; irank++) {
         mpi_state->prof_ranks[irank] = irank;
      }
      mpi_state->my_rank_in_prof = true;
   } else {
      mpi_state->prof_ranks = vftr_expand_rangelist(rangestr, &(mpi_state->nprof_ranks));
      int idx = vftr_binary_search_int(mpi_state->nprof_ranks,
                                       mpi_state->prof_ranks,
                                       process.processID);
      mpi_state->my_rank_in_prof = idx >= 0;
   }
   SELF_PROFILE_END_FUNCTION;
}

void vftr_free_profiled_ranks_list(mpi_state_t *mpi_state) {
   SELF_PROFILE_START_FUNCTION;
   if (mpi_state->nprof_ranks > 0) {
      mpi_state->nprof_ranks = 0;
      free(mpi_state->prof_ranks);
      mpi_state->prof_ranks = NULL;
      mpi_state->my_rank_in_prof = false;
   }
   SELF_PROFILE_END_FUNCTION;
}

long long *vftr_get_total_mpi_overhead(stacktree_t stacktree, int nthreads) {
   SELF_PROFILE_START_FUNCTION;
   // accumulate the mpi overhead for each thread separately
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
         overheads_nsec[threadID] += prof->mpiprof.overhead_nsec;
      }
   }
   SELF_PROFILE_END_FUNCTION;
   return overheads_nsec;
}

long long vftr_get_total_collated_mpi_overhead(collated_stacktree_t stacktree) {
   SELF_PROFILE_START_FUNCTION;
   long long overheads_nsec = 0ll;

   int nstacks = stacktree.nstacks;
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack = stacktree.stacks+istack;
      collated_profile_t *prof = &(stack->profile);
      overheads_nsec += prof->mpiprof.overhead_nsec;
   }
   SELF_PROFILE_END_FUNCTION;
   return overheads_nsec;
}

void vftr_print_mpiprofiling(FILE *fp, mpiprofile_t mpiprof) {
   fprintf(fp, "nmsg: %lld/%lld, msgsize: %lld/%lld\n",
           mpiprof.nsendmessages,
           mpiprof.nrecvmessages,
           mpiprof.send_bytes,
           mpiprof.recv_bytes);
}
