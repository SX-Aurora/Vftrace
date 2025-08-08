#include <stdlib.h>

#include "self_profile.h"
#include "profiling.h"
#include "collated_profiling_types.h"
#include "hwprofiling.h"

#include "collated_callprofiling.h"
#ifdef _MPI
#include "mpiprofiling.h"
#endif
#ifdef _OMP
#include "ompprofiling.h"
#endif
#ifdef _CUDA
#include "collated_cudaprofiling.h"
#endif
#ifdef _ACCPROF
#include "collated_accprofiling.h"
#endif

collated_profile_t vftr_new_collated_profile() {
   SELF_PROFILE_START_FUNCTION;
   collated_profile_t profile;
   profile.callprof = vftr_new_collated_callprofiling();
   profile.hwprof = vftr_new_hwprofiling();
#ifdef _MPI
   profile.mpiprof = vftr_new_mpiprofiling();
#endif
#ifdef _CUDA
   profile.cudaprof = vftr_new_collated_cudaprofiling();
#endif
#ifdef _ACCPROF
   profile.accprof = vftr_new_collated_accprofiling();
#endif
   // TODO: Add other profiles
   SELF_PROFILE_END_FUNCTION;
   return profile;
}

collated_profile_t vftr_add_collated_profiles(collated_profile_t profA,
                                              collated_profile_t profB) {
   collated_profile_t profC;
   profC.callprof = vftr_add_collated_callprofiles(profA.callprof, profB.callprof);
#ifdef _MPI
   profC.mpiprof = vftr_add_mpiprofiles(profA.mpiprof, profB.mpiprof);
#endif
#ifdef _OMP
   profC.ompprof = vftr_add_ompprofiles(profA.ompprof, profB.ompprof);
#endif
#ifdef _CUDA
   //profC.cudaprof = vftr_add_cudaprofiles(profA.cudaprof, profB.cudaprof);
#endif
   // TODO: Add other profiles
   profC.hwprof.counters_incl = NULL;
   profC.hwprof.counters_excl = NULL;
   profC.hwprof.observables = NULL;
   return profC;
}

void vftr_collated_profile_free(collated_profile_t* profile_ptr) {
   SELF_PROFILE_START_FUNCTION;
   vftr_collated_callprofiling_free(&(profile_ptr->callprof));
   vftr_hwprofiling_free(&(profile_ptr->hwprof));
#ifdef _MPI
   vftr_mpiprofiling_free(&(profile_ptr->mpiprof));
#endif
#ifdef _CUDA
   vftr_collated_cudaprofiling_free(&(profile_ptr->cudaprof));
#endif
   // TODO: add other profiles
   SELF_PROFILE_END_FUNCTION;
}
