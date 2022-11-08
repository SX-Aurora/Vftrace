#ifndef SAMPLING_H
#define SAMPLING_H

#include "configuration_types.h"
#include "sampling_types.h"
#include "process_types.h"
#include "timer_types.h"
#include "stack_types.h"

sampling_t vftr_new_sampling(config_t config);

void vftr_finalize_sampling(sampling_t *sampling,
                            config_t config,
                            process_t process,
                            time_strings_t timestrings,
                            double runtime);

void vftr_sample_function_entry(sampling_t *sampling, stack_t stack,
                                long long timestamp);

void vftr_sample_function_exit(sampling_t *sampling, stack_t stack,
                               long long timestamp);

void vftr_sample_init_function_exit(sampling_t *sampling, long long timestamp);

#endif
