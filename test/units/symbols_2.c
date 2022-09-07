#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#include <string.h>

#include "self_profile.h"
#include "symbol_types.h"
#include "symbols.h"
#include "misc_utils.h"

#ifdef _MPI
#include <mpi.h>
#endif

int testfunction(int a) {
   return (a+a)/2;
}

int main(int argc, char **argv) {
   INIT_SELF_PROF_VFTRACE;

#if defined(_MPI)
   PMPI_Init(&argc, &argv);
#else
   (void) argc;
   (void) argv;
#endif

   symboltable_t symboltable = vftr_read_symbols();
   vftr_print_symbol_table(stdout, symboltable);

   // get the address of the function pointer
   uintptr_t func_ptr = (uintptr_t) &testfunction;
   char *func_name = "testfunction";

   // search for the address in the symbol table
   bool address_matches = false;

   bool name_matches = false;
   fprintf(stdout, "searching for (0x%lx) \"%s\"\n",
           func_ptr, func_name);
   for (unsigned int isymb=0; isymb<symboltable.nsymbols; isymb++) {
      fprintf(stdout, "(0x%lx) %s",
              symboltable.symbols[isymb].addr,
              symboltable.symbols[isymb].name);
      if (func_ptr == symboltable.symbols[isymb].addr) {
         address_matches = true;
         if ( !strcmp(func_name, symboltable.symbols[isymb].name)) {
            name_matches = true;
            fprintf(stdout, "   => matches");
         }
      }
      fprintf(stdout, "\n");
   }

   if (!address_matches) {
      fprintf(stdout, "could not find \"%s\" address 0x%lx in symbol table",
              func_name, func_ptr);
   } else {
      if (!name_matches) {
         fprintf(stdout, "name \"%s\" does not match "
                 "for address 0x%lx in symbol table\n",
                 func_name, func_ptr);
      }
   }

#ifdef _MPI
   PMPI_Finalize();
#endif

   FINALIZE_SELF_PROF_VFTRACE;
   return address_matches && name_matches ? 0 : 1;
}
