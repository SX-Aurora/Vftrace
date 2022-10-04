#include <stdlib.h>
#include "cuptiprofiling_types.h"

cuptiprofile_t vftr_new_cuptiprofiling() {
  cuptiprofile_t prof;
  //prof.cupti_object_name = "unknown";
  prof.n_calls = 0;
  prof.t_compute = 0;
  prof.t_memcpy = 0;
  prof.copied_bytes = 0;
  return prof;
}

void vftr_cuptiprofiling_free(cuptiprofile_t *prof_ptr) {
  (void)prof_ptr;
}
