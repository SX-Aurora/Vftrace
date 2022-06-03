#ifndef START_TOOL_H
#define START_TOOL_H

#include <omp.h>
#include <omp-tools.h>

ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version);

#endif
