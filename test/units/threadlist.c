#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "symbol_types.h"
#include "symbols.h"
#include "stack_types.h"
#include "stacks.h"
#include "threads.h"

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
   int func1_idx = 0;
   name = vftr_get_name_from_address(symboltable, addrs+0);
   int func2_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, addrs+0, false);
   name = vftr_get_name_from_address(symboltable, addrs+1);
   int func3_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, addrs+1, false);
   name = vftr_get_name_from_address(symboltable, addrs+2);
   int func4_idx = vftr_new_stack(func3_idx, &stacktree,
                                  name, name, addrs+2, false);
   name = vftr_get_name_from_address(symboltable, addrs+3);
   int func5_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, name, addrs+3, false);
   name = vftr_get_name_from_address(symboltable, addrs+4);
   int func6_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, name, addrs+4, false);
   name = vftr_get_name_from_address(symboltable, addrs+5);
   int func7_idx = vftr_new_stack(func6_idx, &stacktree,
                                  name, name, addrs+5, false);

   threadtree_t threadtree = vftr_new_threadtree();
   int thread0_idx = 0;
   int thread1_idx = vftr_new_thread(thread0_idx, &threadtree);
   int thread2_idx = vftr_new_thread(thread0_idx, &threadtree);
   int thread3_idx = vftr_new_thread(thread2_idx, &threadtree);
   int thread4_idx = vftr_new_thread(thread0_idx, &threadtree);
   int thread5_idx = vftr_new_thread(thread3_idx, &threadtree);
   int thread6_idx = vftr_new_thread(thread4_idx, &threadtree);
   int thread7_idx = vftr_new_thread(thread2_idx, &threadtree);
   int thread8_idx = vftr_new_thread(thread1_idx, &threadtree);
   int thread9_idx = vftr_new_thread(thread0_idx, &threadtree);

   vftr_print_threadlist(stdout, threadtree);

   free_dummy_symbol_table(&symboltable);
   vftr_threadtree_free(&threadtree);
   vftr_stacktree_free(&stacktree);
#ifdef _MPI
   PMPI_Finalize();
#endif

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
