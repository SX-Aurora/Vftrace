#include <stdlib.h>

#include "cuptiprofiling_types.h"

cuptiprofile_t vftr_new_cuptiprofiling() {
  cuptiprofile_t prof;
  prof.events = NULL;
  cudaEventCreate (&(prof.start));
  cudaEventCreate (&(prof.stop));
  return prof;
}

void vftr_cuptiprofiling_free(cuptiprofile_t *prof_ptr) {
  (void)prof_ptr;
}
