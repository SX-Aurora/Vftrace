#ifndef SAMPLING_H
#define SAMPLING_H

#include "environment_types.h"
#include "sampling_types.h"
#include "process_types.h"

sampling_t vftr_new_sampling(environment_t environment);

void vftr_finalize_sampling(sampling_t *sampling,
                            environment_t environment,
                            process_t process);

#endif
