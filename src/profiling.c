#include <stdlib.h>

#include "self_profile.h"
#include "vftrace_state.h"
#include "thread_types.h"
#include "profiling_types.h"
#include "realloc_consts.h"

#include "callprofiling.h"
#ifdef _MPI
#include "mpiprofiling.h"
#endif
#ifdef _OMP
#include "ompprofiling.h"
#endif
#ifdef _CUDA
#include "cudaprofiling.h"
#endif
#ifdef _ACCPROF
#include "accprofiling.h"
#endif
#ifdef _PAPI_AVAIL
#include "hwprofiling.h"
#endif

void vftr_profilelist_realloc(profilelist_t *profilelist_ptr) {
   SELF_PROFILE_START_FUNCTION;
   profilelist_t profilelist = *profilelist_ptr;
   while (profilelist.nprofiles > profilelist.maxprofiles) {
      int maxprofiles = profilelist.maxprofiles*vftr_realloc_rate+vftr_realloc_add;
      profilelist.profiles = (profile_t*)
         realloc(profilelist.profiles, maxprofiles*sizeof(profile_t));
      profilelist.maxprofiles = maxprofiles;
   }
   *profilelist_ptr = profilelist;
   SELF_PROFILE_END_FUNCTION;
}

profile_t vftr_new_profile(int threadID) {
   SELF_PROFILE_START_FUNCTION;
   profile_t profile;
   profile.threadID = threadID;
   profile.callprof = vftr_new_callprofiling();
#ifdef _MPI
   profile.mpiprof = vftr_new_mpiprofiling();
#endif
#ifdef _OMP
   profile.ompprof = vftr_new_ompprofiling();
#endif
#ifdef _CUDA
   profile.cudaprof = vftr_new_cudaprofiling();
#endif
#ifdef _ACCPROF
   profile.accprof = vftr_new_accprofiling();
#endif 
#ifdef _PAPI_AVAIL
   profile.hwprof = vftr_new_hwprofiling();
#endif
   // Add further profiles here
   SELF_PROFILE_END_FUNCTION;
   return profile;
}

int vftr_new_profile_in_list(int threadID, profilelist_t *profilelist_ptr) {
   SELF_PROFILE_START_FUNCTION;
   int profID = profilelist_ptr->nprofiles;
   profilelist_ptr->nprofiles++;
   vftr_profilelist_realloc(profilelist_ptr);

   // insert the new profile in such a way that they are sorted by their threadIDs
   for (int iprof=(profID-1); iprof>=0; iprof--) {
      if (threadID < profilelist_ptr->profiles[iprof].threadID) {
         profilelist_ptr->profiles[iprof+1] = profilelist_ptr->profiles[iprof];
         profID--;
      } else {
         break;
      }
   }

   profile_t *profile = profilelist_ptr->profiles+profID;
   *profile = vftr_new_profile(threadID);

   SELF_PROFILE_END_FUNCTION;
   return profID;
}

profile_t vftr_add_profiles(profile_t profA, profile_t profB) {
   profile_t profC;
   profC.callprof = vftr_add_callprofiles(profA.callprof, profB.callprof);
#ifdef _MPI
   profC.mpiprof = vftr_add_mpiprofiles(profA.mpiprof, profB.mpiprof);
#endif
#ifdef _OMP
   profC.ompprof = vftr_add_ompprofiles(profA.ompprof, profB.ompprof);
#endif
#ifdef _CUDA
   profC.cudaprof = vftr_add_cudaprofiles(profA.cudaprof, profB.cudaprof);
#endif
#ifdef _ACCPROF
   profC.accprof = vftr_add_accprofiles(profA.accprof, profB.accprof);
#endif
   // Add further profiles here.
   return profC;
}

void vftr_profile_free(profile_t* profiles_ptr, int profID) {
   SELF_PROFILE_START_FUNCTION;
   profile_t *profile_ptr = profiles_ptr+profID;
   vftr_callprofiling_free(&(profile_ptr->callprof));
#ifdef _MPI
   vftr_mpiprofiling_free(&(profile_ptr->mpiprof));
#endif
#ifdef _OMP
   vftr_ompprofiling_free(&(profile_ptr->ompprof));
#endif
#ifdef _CUDA
   vftr_cudaprofiling_free(&(profile_ptr->cudaprof));
#endif
#ifdef _ACCPROF
   vftr_accprofiling_free (&(profile_ptr->accprof));
#endif
#ifdef _PAPI_AVAIL
   vftr_hwprofiling_free (&(profile_ptr->hwprof));
#endif
   // Add further profiles here.
   SELF_PROFILE_END_FUNCTION;
}

profilelist_t vftr_new_profilelist() {
   SELF_PROFILE_START_FUNCTION;
   profilelist_t profilelist;
   profilelist.nprofiles = 0;
   profilelist.maxprofiles = 0;
   profilelist.profiles = NULL;
   SELF_PROFILE_END_FUNCTION;
   return profilelist;
}

void vftr_profilelist_free(profilelist_t *profilelist_ptr) {
   SELF_PROFILE_START_FUNCTION;
   profilelist_t profilelist = *profilelist_ptr;
   if (profilelist.nprofiles > 0) {
      for (int iprof=0; iprof<profilelist.nprofiles; iprof++) {
         vftr_profile_free(profilelist.profiles, iprof);
      }
      free(profilelist.profiles);
      profilelist.profiles = NULL;
      profilelist.nprofiles = 0;
      profilelist.maxprofiles = 0;
   }
   *profilelist_ptr = profilelist;
   SELF_PROFILE_END_FUNCTION;
}

profile_t *vftr_get_my_profile(vftr_stack_t *stack,
                               thread_t *thread) {
   SELF_PROFILE_START_FUNCTION;
   profilelist_t *profilelist_ptr = &(stack->profiling);
   // search for the profile matrhing the threadID
   // TODO: binary search?
   int profID = -1;
   for (int iprof=0; iprof<profilelist_ptr->nprofiles; iprof++) {
      profile_t *prof = profilelist_ptr->profiles+iprof;
      if (thread->threadID == prof->threadID) {
         profID = iprof;
         break;
      }
   }
   // if no matching profile is found create one
   // and update the profID
   if (profID == -1) {
      profID = vftr_new_profile_in_list(thread->threadID, profilelist_ptr);
   }

   SELF_PROFILE_END_FUNCTION;
   return profilelist_ptr->profiles+profID;
}
