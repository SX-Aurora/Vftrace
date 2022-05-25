#include <omp.h>
#include <omp-tools.h>

#include "omp_state_types.h"
#include "vftrace_state.h"

#include "parallel_begin.h"
#include "parallel_end.h"
#include "implicit_task.h"

int ompt_initialize(ompt_function_lookup_t lookup,
                    int initial_device_num,
                    ompt_data_t *tool_data) {

   // Get the set_callback function pointer
   ompt_set_callback_t ompt_set_callback =
      (ompt_set_callback_t)lookup("ompt_set_callback");

   // register the available callback functions
   vftr_register_ompt_callback_parallel_begin(ompt_set_callback);
   vftr_register_ompt_callback_parallel_end(ompt_set_callback);
   vftr_register_ompt_callback_implicit_task(ompt_set_callback);

   vftrace.omp_state.initialized = true;
   return 1; // success: activates tool
}
