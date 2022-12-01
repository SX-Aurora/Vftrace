#ifndef PAPI_CALCULATOR_H
#define PAPI_CALCULATOR_H

#include "tinyexpr.h"

#define CALC_N_BUILTIN 2

static const char *builtin_symbols[CALC_N_BUILTIN] = {"T", "ONE"};
enum pcalc_types {PCALC_T, PCALC_ONE};

typedef struct {
   int n_variables;
   int n_te_vars;
   int n_observables;
   double *values;
   //double builtin_values[CALC_N_BUILTIN];
   double *builtin_values;
   te_variable *te_vars;
   te_expr **expr; 
} papi_calculator_t;

papi_calculator_t vftr_init_papi_calculator (int n_variables, int n_observables,
                                             char **symbols, char **formulas);

void vftr_set_papi_calculator_counters (papi_calculator_t *calc, long long *values);
void vftr_set_papi_calculator_builtin (papi_calculator_t *calc, int idx, double value);

double vftr_papi_calculator_evaluate (papi_calculator_t calc, int i_observable);

void vftr_print_papi_calculator_state (papi_calculator_t calc);

#endif
