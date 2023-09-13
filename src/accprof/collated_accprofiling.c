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
   prof.on_nranks = 0;
   memset (prof.ncalls, 0, 2 * sizeof(int));
   memset (prof.avg_ncalls, 0, 2 * sizeof(int));
   memset (prof.max_ncalls, 0, 2 * sizeof(int));
   memset (prof.max_on_rank, 0, 2 * sizeof(int));
   memset (prof.min_ncalls, 0, 2 * sizeof(int));
   memset (prof.min_on_rank, 0, 2 * sizeof(int));
   return prof;
}
