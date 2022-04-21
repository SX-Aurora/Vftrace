#include <stdlib.h>

#include <omp.h>
#include <omp-tools.h>

#include "omp_state_types.h"
#include "vftrace_state.h"

#include "initialize.h"
#include "finalize.h"

ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
   omp_state_t omp_state;
   omp_state.omp_version = omp_version;
   omp_state.runtime_version = runtime_version;
   
   // return from dummy calls that are only done to trick the linker
   // to link all of the OMP-callback layers into the executable
   if (omp_version == 0 && runtime_version == NULL) {return NULL;}

   omp_state.tool_started = true;
   omp_state.start_tool_result.initialize = &ompt_initialize;
   omp_state.start_tool_result.finalize = &ompt_finalize;

   vftrace.omp_state = omp_state;
   return &(vftrace.omp_state.start_tool_result);
}
