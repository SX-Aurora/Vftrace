#ifndef FINALIZE_H
#define FINALIZE_H

#include <omp.h>
#include <omp-tools.h>

void ompt_finalize(ompt_data_t *tool_data);

//extern void (*ompt_finalize_ptr)(ompt_data_t*);

#endif
