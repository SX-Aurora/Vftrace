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
   profile_t profile;
   profile.threadID = threadID;
   profile.callProf = vftr_new_callprofiling();
#ifdef _MPI
   profile.mpiProf = vftr_new_mpiprofiling();
#endif
   // TODO: Add other profiles
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

void vftr_profile_free(profile_t* profiles_ptr, int profID) {
   profile_t *profile_ptr = profiles_ptr+profID;
   vftr_callprofiling_free(&(profile_ptr->callProf));
#ifdef _MPI
   vftr_mpiprofiling_free(&(profile_ptr->mpiProf));
#endif
   // TODO: add other profiles
}

profilelist_t vftr_new_profilelist() {
   profilelist_t profilelist;
   profilelist.nprofiles = 0;
   profilelist.maxprofiles = 0;
   profilelist.profiles = NULL;
   return profilelist;
}

void vftr_profilelist_free(profilelist_t *profilelist_ptr) {
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
}

profile_t *vftr_get_my_profile(stack_t *stack,
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
