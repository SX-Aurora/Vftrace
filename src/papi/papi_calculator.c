#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vftrace_state.h"
#include "signal_handling.h"

#include "papi_calculator.h"

papi_calculator_t vftr_init_papi_calculator (int n_observables, char **symbols, char **formulas) {
   papi_calculator_t calc;

   calc.n_variables = vftrace.papi_state.n_counters;
   calc.n_te_vars = calc.n_variables + CALC_N_BUILTIN;
   calc.n_observables = n_observables;
   calc.values = (double*)malloc(calc.n_variables * sizeof(double));
   calc.builtin_values = (double*)malloc(CALC_N_BUILTIN * sizeof(double));
   calc.te_vars = (te_variable*)malloc(calc.n_te_vars * sizeof(te_variable));
   calc.expr = (te_expr**)malloc(n_observables * sizeof(te_expr*));

   memset (calc.builtin_values, 0, CALC_N_BUILTIN * sizeof(double));
   memset (calc.values, 0, calc.n_variables * sizeof(double));

   for (int i = 0; i < calc.n_te_vars; i++) {
      if (i < CALC_N_BUILTIN) {
         calc.te_vars[i].name = strdup(builtin_symbols[i]);
         calc.te_vars[i].address = &calc.builtin_values[i];
      } else {
         calc.te_vars[i].name = strdup(symbols[i - CALC_N_BUILTIN]);
         calc.te_vars[i].address = &calc.values[i - CALC_N_BUILTIN];
      }
      calc.te_vars[i].type = 0;
      calc.te_vars[i].context = NULL;
   }

   int err;
   for (int i = 0; i < n_observables; i++) {
      calc.expr[i] = te_compile (formulas[i], calc.te_vars, calc.n_te_vars, &err);
      if (!calc.expr[i]) {
         fprintf (stderr, "  Vftrace error: Formula could not be compiled:\n");
         fprintf (stderr, "  %s\n", formulas[i]);
         fprintf (stderr, "  %*s^\n", err - 1, "");
         fprintf (stderr, "  Possible reasons: Symbols do not exist, or syntax error.\n");
         vftr_abort(0);
      }
   } 
   return calc;
}

void vftr_set_papi_calculator_counters (papi_calculator_t *calc, long long *values) {
   for (int i = 0; i < calc->n_variables; i++) {
      calc->values[i] = (double)values[i];
   }
}

void vftr_set_papi_calculator_builtin (papi_calculator_t *calc, int idx, double value) {
   calc->builtin_values[idx] = value;
}

double vftr_papi_calculator_evaluate (papi_calculator_t calc, int i_observable) {
   return te_eval(calc.expr[i_observable]); 
}

void vftr_print_papi_calculator_state (papi_calculator_t calc) {
   printf ("PAPI calculator: \n");
   printf ("  %d Variables:\n", calc.n_variables);
   for (int i = CALC_N_BUILTIN; i < calc.n_te_vars; i++) {
      printf ("   name: %s\n", calc.te_vars[i].name);
      printf ("   linked to: %p\n", calc.te_vars[i].address);
      printf ("   value: %.10e\n", calc.values[i - CALC_N_BUILTIN]);
   }
   printf ("Builtin Variables:\n", CALC_N_BUILTIN);
   for (int i = 0; i < CALC_N_BUILTIN; i++) {
      printf ("   name: %s\n", calc.te_vars[i].name);
      printf ("   linked to: %p\n", calc.te_vars[i].address); 
      printf ("   value: %.10e\n", calc.builtin_values[i]);
   }
}
