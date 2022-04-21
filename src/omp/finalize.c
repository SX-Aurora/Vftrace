#include <omp.h>
#include <omp-tools.h>

void ompt_finalize(ompt_data_t *tool_data) {
   (void) tool_data;
}

//void (*ompt_finalize_ptr)(ompt_data_t*) = ompt_finalize;
