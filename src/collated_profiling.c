#include <stdlib.h>

#include "self_profile.h"
#include "collated_profiling_types.h"

#include "callprofiling.h"
#ifdef _MPI
#include "mpiprofiling.h"
#endif

collated_profile_t vftr_new_collated_profile() {
   SELF_PROFILE_START_FUNCTION;
   collated_profile_t profile;
   profile.callProf = vftr_new_callprofiling();
#ifdef _MPI
   profile.mpiProf = vftr_new_mpiprofiling();
#endif
   // TODO: Add other profiles
   SELF_PROFILE_END_FUNCTION;
   return profile;
}

void vftr_collated_profile_free(collated_profile_t* profile_ptr) {
   SELF_PROFILE_START_FUNCTION;
   vftr_callprofiling_free(&(profile_ptr->callProf));
#ifdef _MPI
   vftr_mpiprofiling_free(&(profile_ptr->mpiProf));
#endif
   // TODO: add other profiles
   SELF_PROFILE_END_FUNCTION;
}
