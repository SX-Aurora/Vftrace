#ifndef PAPIPROFILING_H
#define PAPIPROFILING_H

#include <stdbool.h>

#include "papiprofiling_types.h"

papiprofile_t vftr_new_papiprofiling();

void vftr_accumulate_papiprofiling (papiprofile_t *prof, bool is_entry);
#endif
