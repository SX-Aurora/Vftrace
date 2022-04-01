#include <stdlib.h>
#include <stdio.h>

#include "symbol_types.h"
#include "symbols.h"

symboltable_t dummy_symbol_table(int nsymbols, uintptr_t baseaddr) {
   symboltable_t symboltable;
   symboltable.nsymbols = nsymbols;
   symboltable.symbols = (symbol_t*) malloc(nsymbols*sizeof(symbol_t));
   for (unsigned int isym=0; isym<symboltable.nsymbols; isym++) {
      symboltable.symbols[isym].addr = baseaddr+isym;
      symboltable.symbols[isym].index = 0;
      int buffsize = 1+snprintf(NULL, 0, "func%u", isym);
      symboltable.symbols[isym].name =
         (char*) malloc(buffsize*sizeof(char));
      snprintf(symboltable.symbols[isym].name, buffsize, "func%u", isym);
   }
   return symboltable;
}

void free_dummy_symbol_table(symboltable_t *symboltable) {
   for (unsigned int isym=0; isym<symboltable->nsymbols; isym++) {
      free(symboltable->symbols[isym].name);
   }
   free(symboltable->symbols);
}
