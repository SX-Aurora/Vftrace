#ifndef PROFILING_H
#define PROFILING_H

#include <stdlib.h>

#include "thread_types.h"
#include "profiling_types.h"

profile_t vftr_new_profile(int threadID);

int vftr_new_profile_in_list(int threadID, profilelist_t *profilelist_ptr);

void vftr_profile_free(profile_t* profiles_ptr, int profID);

profilelist_t vftr_new_profilelist();

profile_t vftr_add_profiles(profile_t profA, profile_t profB);

void vftr_profilelist_free(profilelist_t *profilelist_ptr);

profile_t *vftr_get_my_profile(stack_t *stack, thread_t *thread);

#endif
