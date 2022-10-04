#include <stdlib.h>

#include "self_profile.h"
#include "collated_profiling_types.h"

#include "collated_callprofiling.h"
#ifdef _MPI
#include "mpiprofiling.h"
#endif
#ifdef _CUPTI
#include "cuptiprofiling.h"
#endif

collated_profile_t vftr_new_collated_profile() {
   SELF_PROFILE_START_FUNCTION;
   collated_profile_t profile;
   profile.callprof = vftr_new_collated_callprofiling();
#ifdef _MPI
   profile.mpiprof = vftr_new_mpiprofiling();
#endif
#ifdef _CUPTI
   profile.cuptiprof = vftr_new_cuptiprofiling();
#endif
   // TODO: Add other profiles
   SELF_PROFILE_END_FUNCTION;
   return profile;
}

void vftr_collated_profile_free(collated_profile_t* profile_ptr) {
   SELF_PROFILE_START_FUNCTION;
   vftr_collated_callprofiling_free(&(profile_ptr->callprof));
#ifdef _MPI
   vftr_mpiprofiling_free(&(profile_ptr->mpiprof));
#endif
#ifdef _CUPTI
   vftr_cuptiprofiling_free(&(profile_ptr->cuptiprof));
#endif
   // TODO: add other profiles
   SELF_PROFILE_END_FUNCTION;
}
