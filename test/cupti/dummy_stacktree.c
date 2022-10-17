#include <stdlib.h>
#include <string.h>

#include "stack_types.h"
#include "stacks.h"
#include "profiling_types.h"
#include "profiling.h"
#include "callprofiling.h"
#include "cuptiprofiling.h"
#include "hashing.h"

static stacktree_t stacktree;
static uintptr_t base_addr = 123456;

void vftr_init_dummy_stacktree (uint64_t t_call) {
   
   stacktree = vftr_new_stacktree();
   char *rootname = "init";
   stacktree.stacks[0].hash = vftr_jenkins_murmur_64_hash (strlen(rootname), (uint8_t*)rootname);
   profile_t *profile = stacktree.stacks[0].profiling.profiles;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, t_call);
}

int get_stack_idx (uint64_t hash) {
   if (hash == 0) return 0;
   for (int i = 0; i < stacktree.nstacks; i++) {
      if (stacktree.stacks[i].hash == hash) return stacktree.stacks[i].lid;
   }
   return -1;
}

// CUPTI is only supported for one OMP thread, so threadID is always 0.
//
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
      int offset = stacktree.nstacks - 1;
      this_stack_idx = vftr_new_stack(stack_idx_orig, &stacktree, stack_top, stack_top,
                                      base_addr + offset, false);
  
      stacktree.stacks[this_stack_idx].hash = new_hash;
   }

   int iprof = 0;
   if (!stack_already_present)
      iprof = vftr_new_profile_in_list(0, &(stacktree.stacks[this_stack_idx]).profiling);
   profile_t *profile = stacktree.stacks[this_stack_idx].profiling.profiles + iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, t_call);
}

void vftr_register_dummy_cupti_stack (char *stackstring, int cbid, float t_ms,
                                      int mem_dir, uint64_t memcpy_bytes) {
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
      int offset = stacktree.nstacks - 1;
      this_stack_idx = vftr_new_stack(stack_idx_orig, &stacktree, stack_top, stack_top,
                                      base_addr + offset, false);
  
      stacktree.stacks[this_stack_idx].hash = new_hash;
   }

   int iprof = 0;
   if (!stack_already_registered)
      iprof = vftr_new_profile_in_list (0, &(stacktree.stacks[this_stack_idx].profiling));
   profile_t *profile = stacktree.stacks[this_stack_idx].profiling.profiles + iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 0);
   vftr_accumulate_cuptiprofiling(&(profile->cuptiprof), cbid, 1, t_ms, mem_dir, memcpy_bytes);
}

stacktree_t vftr_get_dummy_stacktree () {
   vftr_update_stacks_exclusive_time(&stacktree);
   return stacktree;
}
