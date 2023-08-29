#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "self_profile.h"
#include "vftrace_state.h"
#include "vftr_initialize.h"

#ifdef _PAPI_AVAIL
#include <papi.h>
#endif

// main datatype to store everything

vftrace_t vftrace = {
   .hooks = {
      .function_hooks = {
         .enter = &vftr_initialize,
         .exit = NULL
      },
      .prepause_hooks = {
         .enter = NULL,
         .exit = NULL
      }
   },
   .config.valid = false,
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
      .iobuffer_size = 0,
      .iobuffer = NULL,
      .nextsampletime = 0,
      .interval = 0,
      .function_samplecount = 0,
      .message_samplecount = 0,
      .stacktable_offset = 0,
      .samples_offset = 0
   },
   .signal_received = 0,
#ifdef _OMP
   .omp_state = {
      .tool_started = false,
      .initialized = false,
      .omp_version = 0,
      .runtime_version = NULL,
   },
#endif
#ifdef _CUDA
   .cuda_state = {
      .n_devices = 0,
   },
#endif
#ifdef _ACCPROF
   .accprof_state = {
      .n_devices = 0,
      .device_names = NULL,
      .veto_callback_registration = false,
      .open_wait_queues = NULL,
      .n_open_wait_queues = 0,
   },
#endif
   .hwprof_state = {
      .hwc_type = HWC_NONE,
      .active = false,
      .n_counters = 0,
      .counters = NULL,
#ifdef _PAPI_AVAIL
// This initialization is crucial. Otherwise,
// PAPI_create_eventset will fail.
      .papi.eventset = PAPI_NULL,
#else
      .papi.eventset = 0,
#endif
      .veprof.active_counters = NULL,
   },
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
   size -= sizeof(function_hook_t);
   size += vftr_sizeof_function_hook_t(hooks.prepause_hooks);
   return size;
}

unsigned long long vftr_sizeof_config_struct_defaults(char *name) {
   if (name != NULL) {
      return strlen(name);
   }
   return 0;
}

unsigned long long vftr_sizeof_config_bool_t(config_bool_t cfg_bool) {
   unsigned long long size = sizeof(config_bool_t);
   size += vftr_sizeof_config_struct_defaults(cfg_bool.name);
   return size;
}

unsigned long long vftr_sizeof_config_int_t(config_int_t cfg_int) {
   unsigned long long size = sizeof(config_int_t);
   size += vftr_sizeof_config_struct_defaults(cfg_int.name);
   return size;
}

unsigned long long vftr_sizeof_config_float_t(config_float_t cfg_float) {
   unsigned long long size = sizeof(config_float_t);
   size += vftr_sizeof_config_struct_defaults(cfg_float.name);
   return size;
}

unsigned long long vftr_sizeof_config_string_t(config_string_t cfg_string) {
   unsigned long long size = sizeof(config_string_t);
   size += vftr_sizeof_config_struct_defaults(cfg_string.name);
   if (cfg_string.value != NULL) {
      size += strlen(cfg_string.value);
   }
   return size;

}

unsigned long long vftr_sizeof_config_string_list_t (config_string_list_t cfg_string_list) {
   unsigned long long size = sizeof(config_string_list_t);
   size += vftr_sizeof_config_struct_defaults(cfg_string_list.name); 
   size += sizeof(int);
   if (cfg_string_list.n_elements > 0) {
      for (int i = 0; i < cfg_string_list.n_elements; i++) {
         size += strlen(cfg_string_list.values[i]) * sizeof(char);
      }
      size += cfg_string_list.n_elements * sizeof(int);  // sizeof(list_idx);
   }
   return size;
}

unsigned long long vftr_sizeof_config_regex_t(config_regex_t cfg_regex) {
   unsigned long long size = sizeof(config_regex_t);
   size += vftr_sizeof_config_struct_defaults(cfg_regex.name);
   if (cfg_regex.value != NULL) {
      size += strlen(cfg_regex.value);
   }
   if (cfg_regex.regex != NULL) {
      size += sizeof(regex_t);
   }
   return size;
}

unsigned long long vftr_sizeof_config_sort_table_t(config_sort_table_t cfg_sort_table) {
   unsigned long long size = sizeof(config_sort_table_t);
   size += vftr_sizeof_config_struct_defaults(cfg_sort_table.name);
   size -= sizeof(config_string_t);
   size += vftr_sizeof_config_string_t(cfg_sort_table.column);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(cfg_sort_table.ascending);
   return size;
}

unsigned long long vftr_sizeof_config_profile_table_t(config_profile_table_t
                                                      cfg_profile_table) {
   unsigned long long size = sizeof(config_profile_table_t);
   size += vftr_sizeof_config_struct_defaults(cfg_profile_table.name);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(cfg_profile_table.show_table);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(cfg_profile_table.show_calltime_imbalances);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(cfg_profile_table.show_callpath);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(cfg_profile_table.show_overhead);
   size -= sizeof(config_sort_table_t);
   size += vftr_sizeof_config_sort_table_t(cfg_profile_table.sort_table);
   return size;
}

unsigned long long vftr_sizeof_config_name_grouped_profile_table_t(
   config_name_grouped_profile_table_t cfg_profile_table) {
   unsigned long long size = sizeof(config_name_grouped_profile_table_t);
   size += vftr_sizeof_config_struct_defaults(cfg_profile_table.name);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(cfg_profile_table.show_table);
   size -= sizeof(config_int_t);
   size += vftr_sizeof_config_int_t(cfg_profile_table.max_stack_ids);
   size -= sizeof(config_sort_table_t);
   size += vftr_sizeof_config_sort_table_t(cfg_profile_table.sort_table);
   return size;
}

unsigned long long vftr_sizeof_config_sampling_t(config_sampling_t cfg_sampling) {
   unsigned long long size = sizeof(config_sampling_t);
   size += vftr_sizeof_config_struct_defaults(cfg_sampling.name);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(cfg_sampling.active);
   size -= sizeof(config_float_t);
   size += vftr_sizeof_config_float_t(cfg_sampling.sample_interval);
   size -= sizeof(config_int_t);
   size += vftr_sizeof_config_int_t(cfg_sampling.outbuffer_size);
   size -= sizeof(config_regex_t);
   size += vftr_sizeof_config_regex_t(cfg_sampling.precise_functions);
   return size;
}

unsigned long long vftr_sizeof_config_mpi_t(config_mpi_t cfg_mpi) {
   unsigned long long size = sizeof(config_mpi_t);
   size += vftr_sizeof_config_struct_defaults(cfg_mpi.name);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(cfg_mpi.show_table);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(cfg_mpi.log_messages);
   size -= sizeof(config_string_t);
   size += vftr_sizeof_config_string_t(cfg_mpi.only_for_ranks);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(cfg_mpi.show_sync_time);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(cfg_mpi.show_callpath);
   size -= sizeof(config_sort_table_t);
   size += vftr_sizeof_config_sort_table_t(cfg_mpi.sort_table);
   return size;
}

unsigned long long vftr_sizeof_config_cuda_t(config_cuda_t cfg_cuda) {
   unsigned long long size = sizeof(config_cuda_t);
   size += vftr_sizeof_config_struct_defaults(cfg_cuda.name);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(cfg_cuda.show_table);
   size -= sizeof(config_sort_table_t);
   size += vftr_sizeof_config_sort_table_t(cfg_cuda.sort_table);
   return size;
}

unsigned long long vftr_sizeof_config_accprof_t(config_accprof_t cfg_accprof) {
   unsigned long long size = sizeof(config_accprof_t);
   size += vftr_sizeof_config_struct_defaults(cfg_accprof.name);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(cfg_accprof.show_table);
   size -= sizeof(config_sort_table_t);
   size += vftr_sizeof_config_sort_table_t(cfg_accprof.sort_table);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(cfg_accprof.show_event_details);
   return size;
}

unsigned long long vftr_sizeof_config_hwcounters_t (config_hwcounters_t cfg_hwc) {
  unsigned long long size = sizeof(config_hwcounters_t);
  size += vftr_sizeof_config_struct_defaults (cfg_hwc.name);
  size -= sizeof(config_string_list_t);
  size += vftr_sizeof_config_string_list_t (cfg_hwc.hwc_name);
  size -= sizeof(config_string_list_t);
  size += vftr_sizeof_config_string_list_t (cfg_hwc.symbol);
  return size;
}

unsigned long long vftr_sizeof_config_hwobservables_t (config_hwobservables_t cfg_hwo) {
  unsigned long long size = sizeof(config_hwobservables_t);
  size += vftr_sizeof_config_struct_defaults (cfg_hwo.name);
  size -= sizeof(config_string_list_t);
  size += vftr_sizeof_config_string_list_t (cfg_hwo.obs_name);
  size -= sizeof(config_string_list_t);
  size += vftr_sizeof_config_string_list_t (cfg_hwo.formula_expr);
  size -= sizeof(config_string_list_t);
  size += vftr_sizeof_config_string_list_t (cfg_hwo.unit);
  return size;
}

unsigned long long vftr_sizeof_config_hwprof_t (config_hwprof_t cfg_hwprof) {
   unsigned long long size = sizeof(config_hwprof_t);
   size += vftr_sizeof_config_struct_defaults (cfg_hwprof.name);
   size -= sizeof(config_bool_t); 
   size += vftr_sizeof_config_bool_t(cfg_hwprof.active);
   size -= sizeof(config_bool_t); 
   size += vftr_sizeof_config_bool_t(cfg_hwprof.show_observables);
   size -= sizeof(config_bool_t); 
   size += vftr_sizeof_config_bool_t(cfg_hwprof.show_counters);
   size -= sizeof(config_int_t);
   size += vftr_sizeof_config_bool_t(cfg_hwprof.show_summary);
   size -= sizeof(config_bool_t); 
   size += vftr_sizeof_config_int_t(cfg_hwprof.sort_by_column);
   size -= sizeof(config_hwcounters_t);
   size += vftr_sizeof_config_hwcounters_t (cfg_hwprof.counters);
   size -= sizeof(config_hwobservables_t);
   size += vftr_sizeof_config_hwobservables_t (cfg_hwprof.observables);
   return size;
}

unsigned long long vftr_sizeof_config_t(config_t config) {
   unsigned long long size = sizeof(config_t);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(config.off);
   size -= sizeof(config_string_t);
   size += vftr_sizeof_config_string_t(config.output_directory);
   size -= sizeof(config_string_t);
   size += vftr_sizeof_config_string_t(config.outfile_basename);
   size -= sizeof(config_string_t);
   size += vftr_sizeof_config_string_t(config.logfile_for_ranks);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(config.print_config);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(config.strip_module_names);
   size -= sizeof(config_bool_t);
   size += vftr_sizeof_config_bool_t(config.demangle_cxx);
   size -= sizeof(config_profile_table_t);
   size += vftr_sizeof_config_profile_table_t(config.profile_table);
   size -= sizeof(config_name_grouped_profile_table_t);
   size += vftr_sizeof_config_name_grouped_profile_table_t(config.name_grouped_profile_table);
   size -= sizeof(config_sampling_t);
   size += vftr_sizeof_config_sampling_t(config.sampling);
   size -= sizeof(config_mpi_t);
   size += vftr_sizeof_config_mpi_t(config.mpi);
   size -= sizeof(config_cuda_t);
   size += vftr_sizeof_config_cuda_t(config.cuda);
   size -= sizeof(config_hwprof_t);
   size += vftr_sizeof_config_hwprof_t(config.hwprof);
   if (config.config_file_path != NULL) {
      size += strlen(config.config_file_path);
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

#ifdef _CUDA
unsigned long long vftr_sizeof_cudaprofile_t(cudaprofile_t cudaprof) {
   return sizeof(cudaprofile_t);
}

unsigned long long vftr_sizeof_collated_cudaprofile_t(collated_cudaprofile_t cudaprof) {
   return sizeof(collated_cudaprofile_t);
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
#ifdef _CUDA
   size -= sizeof(cudaprofile_t);
   size += vftr_sizeof_cudaprofile_t(profile.cudaprof);
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
#ifdef _CUDA
   size -= sizeof(collated_cudaprofile_t);
   size += vftr_sizeof_collated_cudaprofile_t(profile.cudaprof);
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

unsigned long long vftr_sizeof_stack_t(vftr_stack_t stack) {
   unsigned long long size = sizeof(vftr_stack_t);
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
   size += (stacktree.maxstacks - stacktree.nstacks) * sizeof(vftr_stack_t);
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
   unsigned long long size = sizeof(sampling_t);
   size += sampling.iobuffer_size;
   return size;
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

#ifdef _CUDA
unsigned long long vftr_sizeof_cuda_state_t(cuda_state_t cuda_state) {
   return sizeof(cuda_state_t);
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
   size -= sizeof(config_t);
   size += vftr_sizeof_config_t(vftrace_state.config);
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
#ifdef _CUDA
   size -= sizeof(cuda_state_t);
   size += vftr_sizeof_cuda_state_t(vftrace_state.cuda_state);
#endif
   size -= sizeof(vftr_size_t);
   size += vftr_sizeof_vftr_size_t(vftrace_state.size);
   SELF_PROFILE_END_FUNCTION;
   return size;
}
