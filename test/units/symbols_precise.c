#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#include <string.h>

#include "self_profile.h"
#include "symbol_types.h"
#include "symbols.h"
#include "misc_utils.h"
#include "regular_expressions.h"

#include "protofuncts.h"

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
   regex_t *testregex = vftr_compile_regexp("^pfunc_[46]_1[0-9]$");

   symboltable_t symboltable = vftr_read_symbols();
   vftr_symboltable_determine_preciseness(&symboltable,
                                          testregex);
   vftr_print_symbol_table(stdout, symboltable);

   // call the protofunction
   // to force linking the function tree
   pfunc_0_0(1);
 
   // check if all protofunction symbols
   // appear in the symbol table
   bool all_symbs_found = true;
   bool preciseness_correct = true;
   char *funcname = "pfunc";
   int lmax = LMAX;
   int lmax_digits = vftr_count_base_digits((long long) lmax, 10);
   for (int l = 0; l <= lmax; l++) {
      int mmax = 1 << l;
      int mmax_digits = vftr_count_base_digits((long long) mmax, 10);
      // construct the function name
      int func_str_len = strlen(funcname) + lmax_digits + mmax_digits + 3;
      char *func_str = (char*) malloc(func_str_len*sizeof(char));
      for (int m = 0; m < mmax; m++) {
         snprintf(func_str, func_str_len, "%s_%d_%d", funcname, l, m);
         // search for the function in the symbol table
         int idx = -1;
         for (unsigned int isymb=0; isymb<symboltable.nsymbols; isymb++) {
            if (!strcmp(func_str, symboltable.symbols[isymb].name)) {
               idx = isymb;
               break;
            }
         }
         // check if a symbol was missed
         if (idx == -1) {
            fprintf(stdout, "Symbol %s not found\n", func_str);
            all_symbs_found = false;
         }
 
         // check if the preciseness was determined properly
         bool preciseness;
         preciseness = (l == 4) || (l == 6);
         preciseness = preciseness && m >= 10 && m < 20;
         if (preciseness != symboltable.symbols[idx].precise) {
            fprintf(stdout, "Wrong preciseness for symbol %s: %s (expected %s)\n",
                    func_str,
                    symboltable.symbols[idx].precise ? "True" : "False",
                    preciseness ? "True" : "False");
            preciseness_correct = false;
         }
      }
      free(func_str);
   }

   vftr_free_regexp(testregex);
   //regfree(testregex);
   //free(testregex);

#ifdef _MPI
   PMPI_Finalize();
#endif

   bool result = all_symbs_found && preciseness_correct;
   FINALIZE_SELF_PROF_VFTRACE;
   return result ? 0 : 1;
}
