#include <stdlib.h>

#include "self_profile.h"
#include "profiling.h"
#include "collated_profiling_types.h"

#include "collated_callprofiling.h"
#ifdef _MPI
#include "mpiprofiling.h"
#endif
#ifdef _OMP
#include "ompprofiling.h"
#endif
#ifdef _CUDA
#include "cudaprofiling.h"
#endif
#ifdef _VEDA
#include "vedaprofiling.h"
#endif

collated_profile_t vftr_new_collated_profile() {
   SELF_PROFILE_START_FUNCTION;
   collated_profile_t profile;
   profile.callprof = vftr_new_collated_callprofiling();
#ifdef _MPI
   profile.mpiprof = vftr_new_mpiprofiling();
#endif
#ifdef _CUDA
   profile.cudaprof = vftr_new_cudaprofiling();
#endif
#ifdef _VEDA
   profile.vedaprof = vftr_new_vedaprofiling();
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
   profC.cudaprof = vftr_add_cudaprofiles(profA.cudaprof, profB.cudaprof);
#endif
#ifdef _VEDA
   profC.vedaprof = vftr_add_vedaprofiles(profA.vedaprof, profB.vedaprof);
#endif
   // TODO: Add other profiles
   return profC;
}

void vftr_collated_profile_free(collated_profile_t* profile_ptr) {
   SELF_PROFILE_START_FUNCTION;
   vftr_collated_callprofiling_free(&(profile_ptr->callprof));
#ifdef _MPI
   vftr_mpiprofiling_free(&(profile_ptr->mpiprof));
#endif
#ifdef _CUDA
   vftr_cudaprofiling_free(&(profile_ptr->cudaprof));
#endif
#ifdef _VEDA
   vftr_vedaprofiling_free(&(profile_ptr->vedaprof));
#endif
   // TODO: add other profiles
   SELF_PROFILE_END_FUNCTION;
}
