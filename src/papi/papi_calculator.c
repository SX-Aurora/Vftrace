#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi_calculator.h"

papi_calculator_t vftr_init_papi_calculator (int n_variables, int n_observables,
                                             char **symbols, char **formulas) {
   papi_calculator_t calc;

   calc.n_variables = n_variables;
   calc.n_observables = n_observables;
   calc.values = (double*)malloc(n_variables * sizeof(double));
   calc.te_vars = (te_variable*)malloc(n_variables * sizeof(te_variable));
   calc.expr = (te_expr**)malloc(n_observables * sizeof(te_expr*));

   memset (calc.values, 0, n_variables * sizeof(double));
   for (int i = 0; i < n_variables; i++) {
      calc.te_vars[i].name = strdup(symbols[i]);
      calc.te_vars[i].address = &calc.values[i];
      calc.te_vars[i].type = 0;
      calc.te_vars[i].context = NULL;
   }

   int err;
   for (int i = 0; i < n_observables; i++) {
      calc.expr[i] = te_compile (formulas[i], calc.te_vars, n_variables, &err);
      if (!calc.expr[i]) {
         printf ("Error compiling formula:\n%s\n", formulas[i]);
         printf ("%*s^\n", err - 1, "");
      }
   } 
   return calc;
}

void vftr_set_papi_calculator_counters (papi_calculator_t *calc, long long *values) {
   for (int i = 0; i < calc->n_variables - N_BUILTIN_VARIABLES; i++) {
      calc->values[i] = (double)values[i];
   }
}

void vftr_set_papi_calculator_builtins (papi_calculator_t *calc, double T) {
   calc->values[calc->n_variables - N_BUILTIN_VARIABLES] = T;
   calc->values[calc->n_variables - N_BUILTIN_VARIABLES + 1] = 1.0;
}

double vftr_papi_calculator_evaluate (papi_calculator_t calc, int i_observable) {
   return te_eval(calc.expr[i_observable]); 
}

void vftr_print_papi_calculator_state (papi_calculator_t calc) {
   printf ("PAPI calculator: \n");
   printf ("  %d Variables:\n", calc.n_variables);
   for (int i = 0; i < calc.n_variables; i++) {
      printf ("   name: %s\n", calc.te_vars[i].name);
      printf ("   linked to: %p\n", calc.te_vars[i].address);
      printf ("   value: %lf\n", calc.values[i]);
   }
   //printf ("  Formulas:\n");
   //for (int i = 0; i < n_observables: i++) {
   //   printf ("   %s\n", 
   //}
}
