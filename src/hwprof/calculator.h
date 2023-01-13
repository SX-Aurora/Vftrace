#ifndef CALCULATOR_H
#define CALCULATOR_H

#include "tinyexpr.h"

#define CALC_N_BUILTIN 2

static const char *builtin_symbols[CALC_N_BUILTIN] = {"T", "CALLS"};
enum pcalc_types {PCALC_T, PCALC_CALLS};

typedef struct {
   int n_variables;
   int n_te_vars;
   int n_observables;
   double *values;
   double *builtin_values;
   te_variable *te_vars;
   te_expr **expr; 
} vftr_calculator_t;

vftr_calculator_t vftr_init_calculator (int n_observables, char **symbols, char **formulas);

void vftr_set_calculator_counters (vftr_calculator_t *calc, long long *values);
void vftr_set_calculator_builtin (vftr_calculator_t *calc, int idx, double value);

double vftr_calculator_evaluate (vftr_calculator_t calc, int i_observable);

void vftr_print_calculator_state (vftr_calculator_t calc);

#endif
