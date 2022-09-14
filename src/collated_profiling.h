#ifndef COLLATED_PROFILING_H
#define COLLATED_PROFILING_H

#include <stdlib.h>

#include "collated_profiling_types.h"

collated_profile_t vftr_new_collated_profile();

void vftr_collated_profile_free(collated_profile_t* profile_ptr);

#endif
