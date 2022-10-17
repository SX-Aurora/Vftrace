#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "environment_types.h"
#include "environment.h"
#include "symbol_types.h"
#include "symbols.h"
#include "stack_types.h"
#include "stacks.h"
#include "collated_stack_types.h"
#include "collate_stacks.h"

#include "dummysymboltable.h"

#ifdef _MPI
#include <mpi.h>
#endif

int main(int argc, char **argv) {
   INIT_SELF_PROF_VFTRACE;
#if defined(_MPI)
   PMPI_Init(&argc, &argv);
#else
   (void) argc;
   (void) argv;
#endif

   // dummy symboltable
   uintptr_t addrs = 123456;
   symboltable_t symboltable = dummy_symbol_table(6, addrs);

   // build stacktree
   stacktree_t stacktree = vftr_new_stacktree();

   char *name;
   int idx0 = 0;
   // func0<init
   name = vftr_get_name_from_address(symboltable, addrs+0);
   int idx1 = vftr_new_stack(idx0, &stacktree,
                             name, name, addrs+0, false);
   // func2<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+2);
   int idx2 = vftr_new_stack(idx1, &stacktree,
                             name, name, addrs+2, false);

   // func1<init
   name = vftr_get_name_from_address(symboltable, addrs+1);
   int idx3 = vftr_new_stack(idx0, &stacktree,
                             name, name, addrs+1, false);
   // func2<func1<init
   name = vftr_get_name_from_address(symboltable, addrs+2);
   int idx4 = vftr_new_stack(idx3, &stacktree,
                             name, name, addrs+2, false);
   // func3<func1<init
   name = vftr_get_name_from_address(symboltable, addrs+3);
   int idx5 = vftr_new_stack(idx3, &stacktree,
                             name, name, addrs+3, false);
   // func4<func1<init
   name = vftr_get_name_from_address(symboltable, addrs+4);
   int idx6 = vftr_new_stack(idx3, &stacktree,
                             name, name, addrs+4, false);

   // func5<init
   name = vftr_get_name_from_address(symboltable, addrs+5);
   int idx7 = vftr_new_stack(idx0, &stacktree,
                             name, name, addrs+5, false);
   // func3<func5<init
   name = vftr_get_name_from_address(symboltable, addrs+3);
   int idx8 = vftr_new_stack(idx7, &stacktree,
                             name, name, addrs+3, false);
   // func4<func5<init
   name = vftr_get_name_from_address(symboltable, addrs+4);
   int idx9 = vftr_new_stack(idx7, &stacktree,
                             name, name, addrs+4, false);
   // func2<func4<func5<init
   name = vftr_get_name_from_address(symboltable, addrs+2);
   int idx10 = vftr_new_stack(idx9, &stacktree,
                              name, name, addrs+2, false);
   // func3<func4<func5<init
   name = vftr_get_name_from_address(symboltable, addrs+3);
   int idx11 = vftr_new_stack(idx9, &stacktree,
                              name, name, addrs+3, false);

   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);
   vftr_print_collated_stacklist(stdout, collated_stacktree);

   fprintf(stdout, "\n");
   collated_stacktree_t namegrouped_collated_stacktree =
      vftr_collated_stacktree_group_by_name(&collated_stacktree);
   vftr_print_collated_stacklist(stdout, namegrouped_collated_stacktree);

   free_dummy_symbol_table(&symboltable);
   vftr_stacktree_free(&stacktree);
   vftr_collated_stacktree_free(&collated_stacktree);
   vftr_collated_stacktree_free(&namegrouped_collated_stacktree);

#ifdef _MPI
   PMPI_Finalize();
#endif

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
