#include <stdlib.h>

#include "vftrace_state.h"
#include "collate_stacks.h"
#include "collate_profiles.h"
#include "collate_vftr_size.h"

void vftr_collate_ranks(vftrace_t *vftrace_ptr) {
   vftrace_ptr->process.collated_stacktree =
      vftr_collate_stacks(&(vftrace_ptr->process.stacktree));

   vftr_collate_profiles(&(vftrace_ptr->process.collated_stacktree),
                         &(vftrace_ptr->process.stacktree));


   vftr_collate_vftr_size(vftrace_ptr);
}
