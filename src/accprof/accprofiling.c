#include <stdlib.h>

#include "accprofiling_types.h"

accprofile_t vftr_new_accprofiling () {
   accprofile_t prof;
   prof.event_type = acc_ev_none;
   prof.copied_bytes = 0;
   prof.var_name = NULL;
   prof.kernel_name = NULL;
   return prof;
}

accprofile_t vftr_add_accprofiles (accprofile_t profA, accprofile_t profB) {
   // Only the amount of bytes moved can differ between the profiles
   accprofile_t profC;
   profC.event_type = profA.event_type; 
   profC.var_name = profA.var_name;
   profC.kernel_name = profA.kernel_name;
   profC.copied_bytes = profA.copied_bytes + profB.copied_bytes;
}

void vftr_accprofiling_free (accprofile_t *prof_ptr) {
   if (prof_ptr->var_name != NULL) free (prof_ptr->var_name);
   if (prof_ptr->kernel_name != NULL) free (prof_ptr->kernel_name);
}
