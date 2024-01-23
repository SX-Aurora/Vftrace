#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vftrace_state.h"
#include "signal_handling.h"

#include "calculator.h"

vftr_calculator_t vftr_init_calculator (int n_observables, char **symbols, char **formulas) {
   vftr_calculator_t calc;

   calc.builtin_values = (double*)malloc(NSYM_BUILTIN * sizeof(double));
   calc.builtin_symbols = (const char**)malloc(NSYM_BUILTIN * sizeof(const char*));
   calc.builtin_symbols[0] = "T";
   calc.builtin_symbols[1] = "CALLS";

   calc.n_variables = vftrace.hwprof_state.n_counters;
   calc.n_te_vars = calc.n_variables + NSYM_BUILTIN;
   calc.n_observables = n_observables;
   calc.values = (double*)malloc(calc.n_variables * sizeof(double));
   calc.te_vars = (te_variable*)malloc(calc.n_te_vars * sizeof(te_variable));
   calc.expr = (te_expr**)malloc(n_observables * sizeof(te_expr*));

   memset (calc.builtin_values, 0, NSYM_BUILTIN * sizeof(double));
   memset (calc.values, 0, calc.n_variables * sizeof(double));

   for (int i = 0; i < calc.n_te_vars; i++) {
      if (i < NSYM_BUILTIN) {
         calc.te_vars[i].name = strdup(calc.builtin_symbols[i]);
         calc.te_vars[i].address = &calc.builtin_values[i];
      } else {
         calc.te_vars[i].name = strdup(symbols[i - NSYM_BUILTIN]);
         calc.te_vars[i].address = &calc.values[i - NSYM_BUILTIN];
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

void vftr_set_calculator_counters (vftr_calculator_t *calc, long long *values) {
   for (int i = 0; i < calc->n_variables; i++) {
      calc->values[i] = (double)values[i];
   }
}

void vftr_set_calculator_builtin (vftr_calculator_t *calc, int idx, double value) {
   calc->builtin_values[idx] = value;
}

double vftr_calculator_evaluate (vftr_calculator_t calc, int i_observable) {
   return te_eval(calc.expr[i_observable]); 
}

void vftr_print_calculator_state (vftr_calculator_t calc) {
   printf ("Vftrace calculator: \n");
   printf ("  %d Variables:\n", calc.n_variables);
   for (int i = NSYM_BUILTIN; i < calc.n_te_vars; i++) {
      printf ("   name: %s\n", calc.te_vars[i].name);
      printf ("   linked to: %p\n", calc.te_vars[i].address);
      printf ("   value: %.10e\n", calc.values[i - NSYM_BUILTIN]);
   }
   printf ("Builtin Variables:\n");
   for (int i = 0; i < NSYM_BUILTIN; i++) {
      printf ("   name: %s\n", calc.te_vars[i].name);
      printf ("   linked to: %p\n", calc.te_vars[i].address); 
      printf ("   value: %.10e\n", calc.builtin_values[i]);
   }
}

void vftr_calculator_free(vftr_calculator_t* calc) {
    if (calc->values) {
        free(calc->values);
        calc->values = NULL;
    }
    if (calc->builtin_values) {
        free(calc->builtin_values);
        calc->builtin_values = NULL;
    }
    if (calc->builtin_symbols) {
        free(calc->builtin_symbols);
        calc->builtin_symbols = NULL;
    }
    if (calc->te_vars) {
        for (int i = 0; i < calc->n_te_vars; i++) {
            free((char*)calc->te_vars[i].name);
            calc->te_vars[i].name = NULL;
        }
        free(calc->te_vars);
        calc->te_vars = NULL;
    }
    if (calc->expr) {
        for (int i = 0; i < calc->n_observables; i++) {
            if (calc->expr[i]) {
                te_free(calc->expr[i]);
                calc->expr[i] = NULL;
            }
        }
        free(calc->expr);
        calc->expr = NULL;
    }
}
