#ifndef ACCPROFILING_H
#define ACCPROFILING_H

#include "accprofiling_types.h"

accprofile_t vftr_new_accprofiling();
accprofile_t vftr_add_accprofiles (accprofile_t profA, accprofile_t profB);
vftr_accprofiling_free (accprofile_t *prof_ptr);

#endif
