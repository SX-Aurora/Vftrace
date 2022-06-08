#ifndef PROFILING_H
#define PROFILING_H

#include <stdlib.h>

#include "thread_types.h"
#include "profiling_types.h"

int vftr_new_profile(int threadID, profilelist_t *profilelist_ptr);

void vftr_profile_free(profile_t* profiles_ptr, int profID);

profilelist_t vftr_new_profilelist();

void vftr_profilelist_free(profilelist_t *profilelist_ptr);

profile_t *vftr_get_my_profile(stack_t *stack, thread_t *thread);

#endif
