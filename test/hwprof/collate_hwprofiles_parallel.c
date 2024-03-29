#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "vftrace_state.h"
#include "stack_types.h"
#include "stacks.h"
#include "hashing.h"
#include "profiling.h"
#include "callprofiling.h"
#include "configuration_types.h"
#include "configuration.h"
#include "stack_types.h"
#include "collated_stack_types.h"
#include "collate_stacks.h"
#include "collate_profiles.h"

#include "hwprofiling.h"
#include "hwprof_logfile.h"
#include "hwprof_ranklogfile.h"

stacktree_t dummy_stacktree;
static uintptr_t base_addr = 123456;

void vftr_init_dummy_stacktree (uint64_t t_call) {
   dummy_stacktree = vftr_new_stacktree();
   char *rootname = "init";
   dummy_stacktree.stacks[0].hash = vftr_jenkins_murmur_64_hash (strlen(rootname), (uint8_t*)rootname);
   profile_t *profile = dummy_stacktree.stacks[0].profiling.profiles;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, t_call);
}

int get_stack_idx (uint64_t hash) {
   if (hash == 0) return 0;
   for (int i = 0; i < dummy_stacktree.nstacks; i++) {
      if (dummy_stacktree.stacks[i].hash == hash) return dummy_stacktree.stacks[i].lid;
   }
   return -1;
}

void vftr_register_dummy_call_stack (char *stackstring, uint64_t t_call) {
   int slen = strlen(stackstring);
   uint64_t new_hash = vftr_jenkins_murmur_64_hash (slen, (uint8_t*)stackstring); 
   int this_stack_idx = get_stack_idx (new_hash);
   bool stack_already_present = this_stack_idx >= 0;
   if (!stack_already_present) {
      char *s = (char*)malloc(slen*sizeof(char));
      strcpy(s, stackstring);
      char *stack_top = strtok(s, "<");
      char *stack_bottom = strtok(NULL, "");

      uint64_t hash_bottom = vftr_jenkins_murmur_64_hash (strlen(stack_bottom), (uint8_t*)stack_bottom);
      int stack_idx_orig = get_stack_idx (hash_bottom);
      int offset = dummy_stacktree.nstacks - 1;
      this_stack_idx = vftr_new_stack(stack_idx_orig, &dummy_stacktree, stack_top, stack_top,
                                      base_addr + offset, false);
  
      dummy_stacktree.stacks[this_stack_idx].hash = new_hash;
   }

   int iprof = 0;
   if (!stack_already_present)
      iprof = vftr_new_profile_in_list(0, &(dummy_stacktree.stacks[this_stack_idx]).profiling);
   profile_t *profile = dummy_stacktree.stacks[this_stack_idx].profiling.profiles + iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, t_call);
}

void vftr_register_dummy_hwprof_stack (char *stackstring, long long *counters) {
   int slen = strlen(stackstring);
   uint64_t new_hash = vftr_jenkins_murmur_64_hash (slen, (uint8_t*)stackstring); 
   int this_stack_idx = get_stack_idx (new_hash);
   bool stack_already_registered = this_stack_idx >= 0;
   if (!stack_already_registered) {
      char *s = (char*)malloc(slen*sizeof(char));
      strcpy(s, stackstring);
      char *stack_top = strtok(s, "<");
      char *stack_bottom = strtok(NULL, "");

      uint64_t hash_bottom = vftr_jenkins_murmur_64_hash (strlen(stack_bottom), (uint8_t*)stack_bottom);
      int stack_idx_orig = get_stack_idx (hash_bottom);
      int offset = dummy_stacktree.nstacks - 1;
      this_stack_idx = vftr_new_stack(stack_idx_orig, &dummy_stacktree, stack_top, stack_top,
                                      base_addr + offset, false);
  
      dummy_stacktree.stacks[this_stack_idx].hash = new_hash;
   }

   int iprof = 0;
   if (!stack_already_registered)
      iprof = vftr_new_profile_in_list (0, &(dummy_stacktree.stacks[this_stack_idx].profiling));
   profile_t *profile = dummy_stacktree.stacks[this_stack_idx].profiling.profiles + iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 0);
   vftr_accumulate_hwprofiling(&(profile->hwprof), counters, false);
}

void vftr_update_dummy_observables () {
   int nstacks = dummy_stacktree.nstacks;
   vftr_stack_t *stacks = dummy_stacktree.stacks; 
   for (int istack = 1; istack < nstacks; istack++) {
      vftr_stack_t *this_stack = stacks + istack;
      for (int iprof = 0; iprof < this_stack->profiling.nprofiles; iprof++) {
         profile_t *this_prof = this_stack->profiling.profiles + iprof;
         hwprofile_t *hwprof = &(this_prof->hwprof);
         hwprof->observables[0] = (double)(hwprof->counters_excl[0] + hwprof->counters_excl[1])/2;
      }
   }
}

stacktree_t vftr_get_dummy_stacktree () {
   vftr_update_stacks_exclusive_time(&dummy_stacktree);
   vftr_update_stacks_exclusive_counters (&dummy_stacktree);
   vftr_update_dummy_observables();
   return dummy_stacktree;
}

int main (int argc, char **argv) {
   PMPI_Init (&argc, &argv);
   int nranks, myrank;
   PMPI_Comm_size (MPI_COMM_WORLD, &nranks);
   if (nranks != 2) {
      fprintf (stderr, "This test requires exactly two ranks, "
               "but was started with %d\n", nranks);
      return 1;
   }
   PMPI_Comm_rank (MPI_COMM_WORLD, &myrank);

   int n_counters = 2;
   int n_observables = 1;
   vftrace.hwprof_state.hwc_type = HWC_DUMMY;
   vftrace.hwprof_state.n_counters = n_counters;
   vftrace.hwprof_state.counters = (vftr_counter_t*)malloc (n_counters * sizeof(vftr_counter_t));
   vftrace.hwprof_state.counters[0].name = "dummy1";
   vftrace.hwprof_state.counters[1].name = "dummy2";
  
   vftrace.hwprof_state.n_observables = n_observables;
   vftrace.hwprof_state.observables = (vftr_observable_t*)malloc(n_observables * sizeof(vftr_counter_t));
   vftrace.hwprof_state.observables[0].name = "dummy_obs";
   vftrace.hwprof_state.observables[0].unit = "dummy_unit";

   vftr_init_dummy_stacktree (10);
    
   if (myrank == 0) {
      vftr_register_dummy_call_stack ("func0<init", 1);
      vftr_register_dummy_call_stack ("hwfunc1<init", 2);
      long long c1[] = {0, 1000};
      vftr_register_dummy_hwprof_stack ("hwfunc1<init", c1);
      long long c2[] = {1000, 500};
      for (int i = 0; i < 3; i++) {
         vftr_register_dummy_hwprof_stack ("hwfunc2<func0<init", c2);
      }
   } else {
      vftr_register_dummy_call_stack ("func0<init", 1);
      vftr_register_dummy_call_stack ("hwfunc1<init", 2);
      long long c1[] = {0, 2000};
      vftr_register_dummy_hwprof_stack ("hwfunc1<init", c1);
      long long c2[] = {500, 1000};
      for (int i = 0; i < 5; i++) {
         vftr_register_dummy_hwprof_stack ("hwfunc2<func0<init", c2);
      }
      //// Additional hwfunc not present on rank 0
      long long c3[] = {1000, 0};
      vftr_register_dummy_hwprof_stack ("hwfunc3<func0<init", c3);
   }
 

   stacktree_t stacktree = vftr_get_dummy_stacktree();
   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);
   vftr_collate_profiles (&collated_stacktree, &stacktree);

   for (int i = 0; i < nranks; i++) {
      if (myrank == i) {
        fprintf (stdout, "Ranklogfile for rank %d: \n", i);
        vftr_write_ranklogfile_hwprof_counter_table(stdout, stacktree, vftrace.config);
      }
      fflush(stdout);
      PMPI_Barrier(MPI_COMM_WORLD);
   }
   if (myrank == 0) {
     fprintf (stdout, "Collated logfile: \n");
     vftr_write_logfile_hwprof_counter_table (stdout, collated_stacktree, vftrace.config);
   }

   PMPI_Finalize();
}
