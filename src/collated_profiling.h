#ifndef COLLATED_PROFILING_H
#define COLLATED_PROFILING_H

#include <stdlib.h>

#include "collated_profiling_types.h"

collated_profile_t vftr_new_collated_profile();

collated_profile_t vftr_add_collated_profiles(collated_profile_t profA,
                                              collated_profile_t profB);

void vftr_collated_profile_free(collated_profile_t* profile_ptr);

#endif
