#include "vftr_scenarios.h"

int main (int argc, char **argv) {

#ifdef _MPI
  PMPI_Init(&argc, &argv);
#endif

  if (argc != 2) {
    printf ("scenarios_1: Missing input file argument.\n");
    return 1;
  }

  FILE *fp_in = fopen (argv[1], "r");
  if (fp_in == NULL) {
    printf ("Scenario file %s not found!\n", argv[1]);
    return 1;
  } 

  vftr_read_scenario_file ("", fp_in);
  vftr_scenario_expr_counter_values[0] = 1.5; // c1
  vftr_scenario_expr_counter_values[1] = 0.5; // c2
  vftr_scenario_expr_counter_values[2] = -1.0; // c3
  vftr_scenario_expr_evaluate_all (0.0, 0ll);
// Test indices:
// 0: sum
// 1: difference
// 2: product
// 3: division
// 4: abs
// 5: exp
// 6: log
// 7: sqrt
// 8: 1e3
  fprintf (stdout, "Check standard operations: \n");
  fprintf (stdout, "%s: %lf\n", vftr_scenario_expr_formulas[0].name, vftr_scenario_expr_formulas[0].value);
  fprintf (stdout, "%s: %lf\n", vftr_scenario_expr_formulas[1].name, vftr_scenario_expr_formulas[1].value);
  fprintf (stdout, "%s: %lf\n", vftr_scenario_expr_formulas[2].name, vftr_scenario_expr_formulas[2].value);
  fprintf (stdout, "%s: %lf\n", vftr_scenario_expr_formulas[3].name, vftr_scenario_expr_formulas[3].value);
  fprintf (stdout, "%s: %lf\n", vftr_scenario_expr_formulas[4].name, vftr_scenario_expr_formulas[4].value);
  fprintf (stdout, "%s: %lf\n", vftr_scenario_expr_formulas[5].name, vftr_scenario_expr_formulas[5].value);
  fprintf (stdout, "%s: %lf\n", vftr_scenario_expr_formulas[6].name, vftr_scenario_expr_formulas[6].value);
  fprintf (stdout, "%s: %lf\n", vftr_scenario_expr_formulas[7].name, vftr_scenario_expr_formulas[7].value);
  fprintf (stdout, "%s: %lf\n", vftr_scenario_expr_formulas[8].name, vftr_scenario_expr_formulas[8].value);
  
  fprintf (stdout, "Check that division by zero is protected: \n");
  vftr_scenario_expr_counter_values[1] = 0.0;
  vftr_scenario_expr_evaluate (3, 0.0, 0ll);
  fprintf (stdout, "%s: %lf\n", vftr_scenario_expr_formulas[3].name, vftr_scenario_expr_formulas[3].value);

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
