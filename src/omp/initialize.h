#ifndef INITIALIZE_H
#define INITIALIZE_H

#include <omp.h>
#include <omp-tools.h>

int ompt_initialize(ompt_function_lookup_t lookup,
                    int initial_device_num,
                    ompt_data_t *tool_data);

//extern int (*ompt_initialize_ptr)(ompt_function_lookup_t, int, ompt_data_t*);

#endif
