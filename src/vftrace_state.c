#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "self_profile.h"
#include "vftrace_state.h"
#include "vftr_initialize.h"
#include "environment.h"

// main datatype to store everything

vftrace_t vftrace = {
   .hooks = {
      .function_hooks = {
         .enter = &vftr_initialize,
         .exit = NULL
      }
   },
   .environment.valid = false,
   .symboltable = {
      .nsymbols = 0,
      .symbols = NULL
   },
   .process = {
      .nprocesses = 1,
      .processID = 0,
      .stacktree = {
         .nstacks = 0,
         .maxstacks = 0,
         .stacks = NULL
      },
      .collated_stacktree = {
         .nstacks = 0,
         .stacks = 0
      }
   },
   .state = uninitialized,
   .sampling = {
      .do_sampling = false,
      .vfdfilename = NULL,
      .vfdfilefp = NULL,
      .iobuffer = NULL,
      .nextsampletime = 0,
      .interval = 0,
      .function_samplecount = 0,
      .message_samplecount = 0,
      .stacktable_offset = 0,
      .samples_offset = 0
   },
#ifdef _OMP
   .omp_state = {
      .tool_started = false,
      .initialized = false,
      .omp_version = 0,
      .runtime_version = NULL,
   },
#endif
#ifdef _CUPTI
   .cupti_state = {
      .n_devices = 0,
   },
#endif
#ifdef _MPI
   .mpi_state = {
      .pcontrol_level = 1,
      .nopen_requests = 0,
      .open_requests = NULL,
      .nprof_ranks = 0,
      .prof_ranks = NULL,
      .my_rank_in_prof = false
   },
#endif
   .timestrings = {
      .start_time = NULL,
      .end_time = NULL
   },
   .size = {
      .rank_wise = 0ll,
      .total = 0ll
   }
};

unsigned long long vftr_sizeof_function_hook_t(function_hook_t function_hooks) {
   (void) function_hooks;
   return sizeof(function_hook_t);
}

unsigned long long vftr_sizeof_hooks_t(hooks_t hooks) {
   unsigned long long size = sizeof(hooks_t);
   // need to subtract the size of the types themselves (padding remains)
   // in order to not double count with the function calls
   size -= sizeof(function_hook_t);
   size += vftr_sizeof_function_hook_t(hooks.function_hooks);
   return size;
}

unsigned long long vftr_sizeof_env_var_t(env_var_t env_var) {
   unsigned long long size = sizeof(env_var_t);
   if (env_var.value_kind == env_string) {
      if (env_var.value.string_val != NULL) {
         size += strlen(env_var.value.string_val)*sizeof(char);
      }
   } else if (env_var.value_kind == env_regex) {
      size += sizeof(regex_t);
   }
   if (env_var.value_string != NULL) {
      size += strlen(env_var.value_string)*sizeof(char);
   }
   size += strlen(env_var.name)*sizeof(char);
   return size;
}

unsigned long long vftr_sizeof_environment_t(environment_t environment) {
   unsigned long long size = sizeof(environment_t);
   for (int ivar=0; ivar<environment.nenv_vars; ivar++) {
      env_var_t *env_var = vftr_get_env_var_ptr_by_idx(&environment, ivar);
      size += vftr_sizeof_env_var_t(*env_var);
   }
   return size;
}

unsigned long long vftr_sizeof_symbol_t(symbol_t symbol) {
   unsigned long long size = sizeof(symbol);
   size += strlen(symbol.name)*sizeof(char);
   size += strlen(symbol.cleanname)*sizeof(char);
   return size;
}

unsigned long long vftr_sizeof_symboltable_t(symboltable_t symboltable) {
   unsigned long long size = sizeof(symboltable_t);
   for (unsigned int isymbol=0; isymbol<symboltable.nsymbols; isymbol++) {
      size += vftr_sizeof_symbol_t(symboltable.symbols[isymbol]);
   }
   return size;
}

unsigned long long vftr_sizeof_callprofile_t(callprofile_t callprof) {
   (void) callprof;
   return sizeof(callprofile_t);
}

unsigned long long vftr_sizeof_collated_callprofile_t(collated_callprofile_t callprof) {
   (void) callprof;
   return sizeof(collated_callprofile_t);
}

#ifdef _MPI
unsigned long long vftr_sizeof_mpiprofile_t(mpiprofile_t mpiprof) {
   (void) mpiprof;
   return sizeof(mpiprofile_t);
}
#endif

#ifdef _CUPTI
unsigned long long vftr_sizeof_cuptiprofile_t(cuptiprofile_t cuptiprof) {
   return sizeof(cuptiprofile_t);
}
#endif

unsigned long long vftr_sizeof_profile_t(profile_t profile) {
   unsigned long long size = sizeof(profile_t);
   size -= sizeof(callprofile_t);
   size += vftr_sizeof_callprofile_t(profile.callprof);
#ifdef _MPI
   size -= sizeof(mpiprofile_t);
   size += vftr_sizeof_mpiprofile_t(profile.mpiprof);
#endif
#ifdef _CUPTI
   size -= sizeof(cuptiprofile_t);
   size += vftr_sizeof_cuptiprofile_t(profile.cuptiprof);
#endif
   return size;
}

unsigned long long vftr_sizeof_collated_profile_t(collated_profile_t profile) {
   unsigned long long size = sizeof(profile_t);
   size -= sizeof(collated_profile_t);
   size += vftr_sizeof_collated_callprofile_t(profile.callprof);
#ifdef _MPI
   size -= sizeof(mpiprofile_t);
   size += vftr_sizeof_mpiprofile_t(profile.mpiprof);
#endif
#ifdef _CUPTI
   size -= sizeof(cuptiprofile_t);
   size += vftr_sizeof_cuptiprofile_t(profile.cuptiprof);
#endif
   return size;
}

unsigned long long vftr_sizeof_profilelist_t(profilelist_t profiling) {
   unsigned long long size = sizeof(profilelist_t);
   // it is important to distinguish between used profiles
   // and unused one. Unused ones have no further nested allocated structures
   // Therefore, a simple sizeof() is enough here
   // Used profiles:
   for (int iprof=0; iprof<profiling.nprofiles; iprof++) {
      size += vftr_sizeof_profile_t(profiling.profiles[iprof]);
   }
   // unused profiles:
   size += (profiling.maxprofiles = profiling.nprofiles) * sizeof(profile_t);
   return size;
}

unsigned long long vftr_sizeof_stack_t(stack_t stack) {
   unsigned long long size = sizeof(stack_t);
   size += strlen(stack.name)*sizeof(char);
   size += strlen(stack.cleanname)*sizeof(char);
   size -= sizeof(profilelist_t);
   size += vftr_sizeof_profilelist_t(stack.profiling);
   return size;
}

unsigned long long vftr_sizeof_stacktree_t(stacktree_t stacktree) {
   unsigned long long size = sizeof(stacktree_t);
   // it is important to distinguish between used stacks
   // and unused one. Unused ones have no further nested allocated structures
   // Therefore, a simple sizeof() is enough here
   // Used stacks:
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      size += vftr_sizeof_stack_t(stacktree.stacks[istack]);
   }
   // unused stacks:
   size += (stacktree.maxstacks - stacktree.nstacks) * sizeof(stack_t);
   return size;
}

unsigned long long vftr_sizeof_gid_list_t(gid_list_t gid_list) {
   return sizeof(gid_list_t) + gid_list.ngids*sizeof(int);
}

unsigned long long vftr_sizeof_collated_stack_t(collated_stack_t stack) {
   unsigned long long size = sizeof(collated_stack_t);
   size -= sizeof(collated_profile_t);
   size += vftr_sizeof_collated_profile_t(stack.profile);
   size -= sizeof(gid_list_t);
   size += vftr_sizeof_gid_list_t(stack.gid_list);
   if (stack.name != NULL) {
      size += strlen(stack.name)*sizeof(char);
   }
   return size;
}

unsigned long long vftr_sizeof_collated_stacktree_t(collated_stacktree_t stacktree) {
   unsigned long long size = sizeof(collated_stacktree_t);
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      size += vftr_sizeof_collated_stack_t(stacktree.stacks[istack]);
   }
   return size;
}

unsigned long long vftr_sizeof_threadstack_t(threadstack_t stack) {
   (void) stack;
   return sizeof(threadstack_t);
}

unsigned long long vftr_sizeof_threadstacklist_t(threadstacklist_t stacklist) {
   unsigned long long size = sizeof(threadstacklist_t);
   // it is important to distinguish between used stacks
   // and unused one. Unused ones have no further nested allocated structures
   // Therefore, a simple sizeof() is enough here
   // Used stacks:
   for (int istack=0; istack<stacklist.nstacks; istack++) {
      size += vftr_sizeof_threadstack_t(stacklist.stacks[istack]);
   }
   // unused stacks:
   size += (stacklist.maxstacks - stacklist.nstacks) * sizeof(threadstack_t);
   return size;
}

unsigned long long vftr_sizeof_thread_t(thread_t thread) {
   unsigned long long size = sizeof(thread_t);
   size -= sizeof(threadstacklist_t);
   size += vftr_sizeof_threadstacklist_t(thread.stacklist);
   size += thread.maxsubthreads*sizeof(int);
   return size;
}

unsigned long long vftr_sizeof_threadtree_t(threadtree_t threadtree) {
   unsigned long long size = sizeof(threadtree_t);
   // it is important to distinguish between used threads
   // and unused one. Unused ones have no further nested allocated structures
   // Therefore, a simple sizeof() is enough here
   // Used threads:
   for (int ithread=0; ithread<threadtree.nthreads; ithread++) {
      size += vftr_sizeof_thread_t(threadtree.threads[ithread]);
   }
   // unused threads:
   size += (threadtree.maxthreads - threadtree.nthreads) * sizeof(thread_t);
   return size;
}

unsigned long long vftr_sizeof_process_t(process_t process) {
   unsigned long long size = sizeof(process_t);
   // need to subtract the size of the types themselves (padding remains)
   // in order to not double count with the function calls
   size -= sizeof(stacktree_t);
   size += vftr_sizeof_stacktree_t(process.stacktree);
   size -= sizeof(collated_stacktree_t);
   size += vftr_sizeof_collated_stacktree_t(process.collated_stacktree);
   size -= sizeof(threadtree_t);
   size += vftr_sizeof_threadtree_t(process.threadtree);
   return size;
}

unsigned long long vftr_sizeof_state_t(state_t state) {
   (void) state;
   return sizeof(state_t);
}

unsigned long long vftr_sizeof_sampling_t(sampling_t sampling) {
   (void) sampling;
   return sizeof(sampling_t);
}

unsigned long long vftr_sizeof_time_strings_t(time_strings_t time_strings) {
   unsigned long long size = sizeof(time_strings_t);
   if (time_strings.start_time != NULL) {
      size += strlen(time_strings.start_time);
   }
   if (time_strings.end_time != NULL) {
      size += strlen(time_strings.end_time);
   }
   return size;
}

#ifdef _OMP
unsigned long long vftr_sizeof_omp_state_t(omp_state_t omp_state) {
   (void) omp_state;
   return sizeof(omp_state_t);
}
#endif

#ifdef _MPI
unsigned long long vftr_sizeof_mpi_state_t(mpi_state_t mpi_state) {
   unsigned long long size = sizeof(mpi_state);
   size += mpi_state.nopen_requests*sizeof(vftr_request_t);
   return size;
}
#endif

#ifdef _CUPTI
unsigned long long vftr_sizeof_cupti_state_t(cupti_state_t cupti_state) {
   return sizeof(cupti_state_t);
}
#endif

unsigned long long vftr_sizeof_vftr_size_t(vftr_size_t size) {
   (void) size;
   return sizeof(vftr_size_t);
}

unsigned long long vftr_sizeof_vftrace_t(vftrace_t vftrace_state) {
   SELF_PROFILE_START_FUNCTION;
   unsigned long long size = sizeof(vftrace_t);
   // need to subtract the size of the types themselves (padding remains)
   // in order to not double count with the function calls
   size -= sizeof(hooks_t);
   size += vftr_sizeof_hooks_t(vftrace_state.hooks);
   size -= sizeof(environment_t);
   size += vftr_sizeof_environment_t(vftrace_state.environment);
   size -= sizeof(symboltable_t);
   size += vftr_sizeof_symboltable_t(vftrace_state.symboltable);
   size -= sizeof(process_t);
   size += vftr_sizeof_process_t(vftrace_state.process);
   size -= sizeof(state_t);
   size += vftr_sizeof_state_t(vftrace_state.state);
   size -= sizeof(sampling_t);
   size += vftr_sizeof_sampling_t(vftrace_state.sampling);
   size -= sizeof(time_strings_t);
   size += vftr_sizeof_time_strings_t(vftrace_state.timestrings);
#ifdef _OMP
   size -= sizeof(omp_state_t);
   size += vftr_sizeof_omp_state_t(vftrace_state.omp_state);
#endif
#ifdef _MPI
   size -= sizeof(mpi_state_t);
   size += vftr_sizeof_mpi_state_t(vftrace_state.mpi_state);
#endif
#ifdef _CUPTI
   size -= sizeof(cupti_state_t);
   size += vftr_sizeof_cupti_state_t(vftrace_state.cupti_state);
#endif
   size -= sizeof(vftr_size_t);
   size += vftr_sizeof_vftr_size_t(vftrace_state.size);
   SELF_PROFILE_END_FUNCTION;
   return size;
}
