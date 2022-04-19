#ifndef OMP_STATE_TYPES_H
#define OMP_STATE_TYPES_H

#include <stdbool.h>

#include <omp.h>
#include <omp-tools.h>

typedef struct {
   bool tool_started;
   bool initialized;
   unsigned int omp_version;
   const char *runtime_version;
   ompt_start_tool_result_t start_tool_result;
} omp_state_t;

#endif
