#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#include <string.h>

#include "self_profile.h"
#include "symbol_types.h"
#include "symbols.h"
#include "misc_utils.h"
#include "regular_expressions.h"

#include "vftrace.h"

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

   // compile a test regular expressions to test preciseness determination
   regex_t *testregex = vftr_compile_regexp("$nothing");

   symboltable_t symboltable = vftr_read_symbols();
   vftr_symboltable_determine_preciseness(&symboltable,
                                          testregex);
   vftr_print_symbol_table(stdout, symboltable);

   // trick the linker 
   // to force linking the pause/resume functions
   // but don't call it, as the instrumentation will mess
   // with the state
   if (argc == -1) {
      vftrace_pause();
      vftrace_resume();
   }

   // check if all protofunction symbols
   // appear in the symbol table
   bool all_symbs_found = true;
   bool preciseness_correct = true;
   // search for the pause function in the symbol table
   int idx = -1;
#define NSTR 2
   char *func_strs[NSTR] = {"vftrace_pause", "vftrace_resume"};
   for (int istr = 0; istr < NSTR; istr++) {
      for (unsigned int isymb = 0; isymb < symboltable.nsymbols; isymb++) {
         if (!strcmp(func_strs[istr], symboltable.symbols[isymb].name)) {
            idx = isymb;
            break;
         }
      }
      // check if a symbol was missed
      if (idx == -1) {
         fprintf(stdout, "Symbol %s not found\n", func_strs[istr]);
         all_symbs_found = false;
      } else {
         if (!symboltable.symbols[idx].precise) {
            fprintf(stdout, "Wrong preciseness for symbol %s: %s\n",
                    func_strs[istr],
                    symboltable.symbols[idx].precise ? "True" : "False");
            preciseness_correct = false;
         }
      }
   }

   vftr_free_regexp (testregex);
   //regfree(testregex);
   //free(testregex);

#ifdef _MPI
   PMPI_Finalize();
#endif

   bool result = all_symbs_found && preciseness_correct;
   FINALIZE_SELF_PROF_VFTRACE;
   return result ? 0 : 1;
}
