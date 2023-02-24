#include "collated_accprofiling_types.h"

collated_accprofile_t vftr_new_collated_accprofiling() {
   collated_accprofile_t prof;
   prof.event_type = 0;
   prof.region_id = 0;
   prof.line_start = 0;
   prof.line_end = 0;
   prof.copied_bytes = 0;
   prof.source_file = NULL;
   prof.func_name = NULL;
   prof.var_name = NULL;
   prof.kernel_name = NULL;
   prof.overhead_nsec = 0;
   return prof;
}
