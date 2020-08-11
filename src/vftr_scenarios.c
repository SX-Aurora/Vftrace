/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "vftr_signals.h"
#include "vftr_fileutils.h"
#include "vftr_hwcounters.h"
#include "vftr_scenarios.h"
#include "jsmn.h"
#include "tinyexpr.h"

#ifdef HAS_VEPERF
#include "veperf.h"
#endif

/**********************************************************************/

void vftr_scenario_print_formula (FILE *fp, function_expr_t formula) {
	fprintf (fp, "Formula name: %s\n", formula.name);
	fprintf (fp, "   Expression: %s\n", formula.formula);
	fprintf (fp, "   Protected values: %s\n", 
		 formula.protected_values ? formula.protected_values : "None");
	fprintf (fp, "   Default value: %lf\n", formula.default_value);
	fprintf (fp, "   Current value: %lf\n", formula.value);
	fprintf (fp, "   Integrated: %s\n", vftr_bool_to_string (formula.integrated));
}

/**********************************************************************/

void vftr_write_scenario_header_to_vfd (FILE *fp) {
	fwrite (&scenario_expr_n_formulas, sizeof(int), 1, fp);
#if defined(HAS_VEPERF) || defined(HAS_PAPI)
	for (int i = 0; i < scenario_expr_n_formulas; i++) {
		fwrite (scenario_expr_formulas[i].name, SCENARIO_NAME_LEN, 1, fp);
		fwrite (&scenario_expr_formulas[i].integrated, sizeof(int), 1, fp);
	}
#endif
}

/**********************************************************************/

void vftr_write_observables_to_vfd (unsigned long long cycles, FILE *fp) {
#if defined(HAS_VEPERF) || defined(HAS_PAPI)
	scenario_expr_evaluate_all (0., cycles);
	for (int i = 0; i < scenario_expr_n_formulas; i++) {
		fwrite (&scenario_expr_formulas[i].value, sizeof(double), 1, fp);
	}
#endif
}

/**********************************************************************/

void trim_trailing_white_spaces (char *s) {
	int last_index = -1;
	int i = 0;
	while (s[i] != '\0') {
		last_index = (s[i] != ' ' && s[i] != '\t' && s[i] != '\n') ? i : last_index;
		i++;
	}
	s[last_index+1] = '\0';
}

/**********************************************************************/
//What follows it an inconveniently ugly and naive json parser

te_variable *te_vars;
te_expr **expr;

int read_counters;
int read_observables;
int read_papi_name;
int read_name;
int read_symbol;
int read_formula;
int read_runtime;
int read_protected;
int read_default;
int read_unit;
int read_group;
int read_column1;
int read_column2;
int read_scenario_name;
int read_spec;

// The user can define a scenario name to be included in the logfile
// TODO: Implement this
char *vftr_scenario_string;

int json_fetch (const char *js, jsmntok_t *tok, size_t count) {
	jsmntok_t *key;
	if (tok->type == JSMN_PRIMITIVE) {
		return 1;
	} else if (tok->type == JSMN_STRING) {
		char *s = strndup(js + tok->start, tok->end - tok->start);
		if (read_papi_name) {
			scenario_expr_counter_names[scenario_expr_n_vars] = strdup(s);
			read_papi_name = 0;
		} else if (read_counters && read_symbol) {
			scenario_expr_vars[scenario_expr_n_vars] = strdup(s);
			read_symbol = 0;
			scenario_expr_n_vars++;
		} else if (read_name) {
			scenario_expr_formulas[scenario_expr_n_formulas++].name = strdup(s);
			scenario_expr_formulas[scenario_expr_n_formulas-1].protected_values = NULL;
			read_name = 0;
		} else if (read_formula) {
			scenario_expr_formulas[scenario_expr_n_formulas-1].formula = strdup(s);		
			read_formula = 0;
		} else if (read_runtime) {
			if (!strcmp (s, "yes")) {
				scenario_expr_formulas[scenario_expr_n_formulas-1].integrated = false;
			}
			read_runtime = 0;	
		} else if (read_protected) {
			scenario_expr_formulas[scenario_expr_n_formulas-1].protected_values = strdup(s);
			read_protected = 0;
		} else if (read_default) {
			sscanf (s, "%lf", &scenario_expr_formulas[scenario_expr_n_formulas-1].default_value);	
			read_default = 0;
		} else if (read_unit) {
			scenario_expr_format[scenario_expr_n_formulas-1].unit = strdup (s);
			read_unit = 0;
		} else if (read_group) {
			scenario_expr_format[scenario_expr_n_formulas-1].group = strdup (s);
			read_group = 0;
		} else if (read_column1) {
			scenario_expr_format[scenario_expr_n_formulas-1].column1 = strdup (s);
			read_column1 = 0;
		} else if (read_column2) {
			scenario_expr_format[scenario_expr_n_formulas-1].column2 = strdup (s);
			read_column2 = 0;
		} else if (read_scenario_name) {
			vftr_scenario_string = strdup (s);
			read_scenario_name = 0;
		} else if (read_spec) {
			char *s1 = strtok (s, ".");
			if (!strcmp (s1, "*")) {
				scenario_expr_format[scenario_expr_n_formulas-1].decpl_1 = 0;
			} else {
				scenario_expr_format[scenario_expr_n_formulas-1].decpl_1 = atoi(s1);
			}
			scenario_expr_format[scenario_expr_n_formulas-1].decpl_2 = atoi(strtok (NULL, ""));
			read_spec = 0;
		}

		if (!strcmp (s, "counters")) {
			read_counters = 1;
			read_observables = 0;
		} else if (!strcmp (s, "observables")) {
			read_observables = 1;
			read_counters = 0;
		} else if (!strcmp (s, "papi_name")) {
			read_papi_name = 1;
		} else if (!strcmp (s, "name")) {
			read_name = 1;
		} else if (!strcmp (s, "symbol")) {
			read_symbol = 1;
		} else if (!strcmp (s, "formula")) {
			read_formula = 1;
		} else if (!strcmp (s, "divide_by_runtime")) {
			read_runtime = 1;
		} else if (!strcmp (s, "protected")) {
			read_protected = 1;
		} else if (!strcmp (s, "default")) {
			read_default = 1;
		} else if (!strcmp (s, "unit")) {
			read_unit = 1;
		} else if (!strcmp (s, "spec")) {
			read_spec = 1;
		} else if (!strcmp (s, "group")) {
			read_group = 1;
		} else if (!strcmp (s, "column1")) {
			read_column1 = 1;
		} else if (!strcmp (s, "column2")) {
			read_column2 = 1;
		} else if (!strcmp (s, "scenario_name")) {
			read_scenario_name = 1;
		}
		return 1;
	} else if (tok->type == JSMN_OBJECT) {
		int j = 0;
		for (int i = 0; i < tok->size; i++) {
			key = tok + j + 1;
			j += json_fetch (js, key, count - j);
			if (key->size > 0) {
				j += json_fetch (js, tok + j + 1, count - j);
			}
		}
		return j + 1;
	} else if (tok->type == JSMN_ARRAY) {
		int j = 0;
		for (int i = 0; i < tok->size; i++) {
			j += json_fetch (js, tok + j + 1, count - j);
		}
		return j + 1;
	}
	return 0;	
}

/**********************************************************************/

int vftr_read_scenario_file (char *filename, FILE *fp_ext) {
	int token_len;
	jsmn_parser p;
	jsmntok_t *token;
	char buf[BUFSIZ];
	size_t token_count = 2;
	char *js = NULL;
	size_t jslen = 0;
	FILE *fp;

	read_counters = 0;
	read_observables = 0;
	read_name = 0;
	read_symbol = 0;
	read_formula = 0;
	read_protected = 0;
	read_default = 0;

	jsmn_init (&p);

	token = malloc (sizeof(*token) * token_count);

// When an external file pointer is given, e.g. for a unit test
// we do not want to close the file in this routine and assume
// that this is done somewhere else.
	bool need_to_close_file = true;
	if (fp_ext == NULL) {
		if ((fp = fopen (filename, "r")) == NULL) {
			printf ("Failed to open scenario file %s\n", filename);
			return -1;
		}
	} else if (fp_ext) {
		fp = fp_ext;	
		need_to_close_file = false;
	} else {
		return -1;
	}
	
	for (int i = 0; i < TE_MAX; i++) {
		scenario_expr_formulas[i].integrated = true;
	}
	
	while ((token_len = fread (buf, 1, sizeof(buf), fp))) { 
		js = realloc (js, jslen + token_len + 1);	
		strncpy (js + jslen, buf, token_len);
		jslen += token_len;
		again:
		token_len = jsmn_parse (&p, js, jslen, token, token_count);
		if (token_len < 0) {
			if (token_len == JSMN_ERROR_NOMEM) {
				token_count *= 2;
				token = realloc (token, sizeof(*token) * token_count);
				if (token == NULL) {
					break;
				}
				goto again;
			}	
		} else {
			json_fetch (js, token, token_count);	
		}
	}
	if (need_to_close_file) fclose (fp);
	
	te_vars = (te_variable *) malloc ((scenario_expr_n_vars + 3) * sizeof (te_variable));
	scenario_expr_counter_values = (double *) malloc (scenario_expr_n_vars * sizeof (double));
	for (int i = 0; i < scenario_expr_n_vars; i++) {
		te_vars[i].name = scenario_expr_vars[i];
		te_vars[i].address = &scenario_expr_counter_values[i];
		te_vars[i].type = 0;
		te_vars[i].context = NULL;
	}	
	te_vars[scenario_expr_n_vars].name = "runtime";
	te_vars[scenario_expr_n_vars].address = &scenario_expr_runtime;
	te_vars[scenario_expr_n_vars].type = 0;
	te_vars[scenario_expr_n_vars].context = NULL;

	te_vars[scenario_expr_n_vars+1].name = "cycles";
	te_vars[scenario_expr_n_vars+1].address = &scenario_expr_cycles;
	te_vars[scenario_expr_n_vars+1].type = 0;
	te_vars[scenario_expr_n_vars+1].context = NULL;

	te_vars[scenario_expr_n_vars+2].name = "cycletime";
	te_vars[scenario_expr_n_vars+2].address = &scenario_expr_cycletime;
	te_vars[scenario_expr_n_vars+2].type = 0;
	te_vars[scenario_expr_n_vars+2].context = NULL;

	int err;
	expr = (te_expr **) malloc (scenario_expr_n_formulas * sizeof (te_expr *));	
	for (int i = 0; i < scenario_expr_n_formulas; i++) {
		expr[i] = te_compile (scenario_expr_formulas[i].formula, te_vars, scenario_expr_n_vars + 3, &err);
		if (!expr[i]) {
			printf ("ERROR COMPILING FORMULA:\n%s\n", scenario_expr_formulas[i].formula);
			printf ("%*s^\n", err - 1, "");
			vftr_abort();
		}
			
	}
	return 0;
}

/**********************************************************************/

int variable_index (char *varname) {
	for (int i = 0; i < scenario_expr_n_vars + 1; i++) {
		if (!strcmp (varname, te_vars[i].name)) {
			return i;
		}
	}
	return -1;
}

/**********************************************************************/

void scenario_expr_evaluate (int i_scenario, double runtime, unsigned long long cycles) {
	scenario_expr_runtime = runtime;
	scenario_expr_cycles = (double)cycles;
	scenario_expr_cycletime = runtime / cycles;

	int i_protected = scenario_expr_formulas[i_scenario].protected_values ?
		variable_index (scenario_expr_formulas[i_scenario].protected_values) : -1;
	double check_value = -1.0;
	if (i_protected < 0) {
		scenario_expr_formulas[i_scenario].value = te_eval (expr[i_scenario]);		
	} else if (i_protected == scenario_expr_n_vars) {
		check_value = scenario_expr_runtime;
	} else if (i_protected == scenario_expr_n_vars + 1) {
		check_value = scenario_expr_runtime;
	} else {
		check_value = (double)scenario_expr_counter_values[i_protected];
	}
	if (check_value == 0.) {
		scenario_expr_formulas[i_scenario].value = scenario_expr_formulas[i_scenario].default_value;	
	} else {
		scenario_expr_formulas[i_scenario].value = te_eval (expr[i_scenario]);
		if (!scenario_expr_formulas[i_scenario].integrated && runtime > 0.) {
			scenario_expr_formulas[i_scenario].value /= runtime;
		}
	}
}

/**********************************************************************/

void scenario_expr_evaluate_all (double runtime, unsigned long long cycles) {
	for (int i = 0; i < scenario_expr_n_formulas; i++) {
		scenario_expr_evaluate (i, runtime, cycles);
	}
}

/**********************************************************************/

static int n_decimal_places (int i_scenario) {
	long long count;
	int k;
	int n_decimal_places = 1;
	for (count = (long long)scenario_expr_formulas[i_scenario].value, k = 0; count; count /= 10, k++);
	if (k > n_decimal_places) n_decimal_places = k;
	return n_decimal_places;
}

/**********************************************************************/
		
static void get_format (char *fmt, int i) {
	int behind_comma = scenario_expr_format[i].decpl_2;
	static int total = 0;
	int tmp = scenario_expr_format[i].decpl_1 > 0 ?
		scenario_expr_format[i].decpl_1 : n_decimal_places(i) + behind_comma;
	total = tmp > total ? tmp : total;
	sprintf (fmt, "%%%d.%dlf ", total, behind_comma);
}

/**********************************************************************/

char *formats[TE_MAX];

void scenario_expr_set_formats () {
	char tmp[10];
	for (int i = 0; i < scenario_expr_n_formulas; i++) {
		// I'm to stupid to use format directly
		get_format (tmp, i);
		formats[i] = strdup(tmp);
	}
}

/**********************************************************************/

#define SUMMARY_LINE_SIZE 27
void scenario_expr_print_summary (FILE *fp) {
	char fmt[10];
	for (int i = 0; i < scenario_expr_n_formulas; i++) {
		int n_chars = strlen(scenario_expr_formulas[i].name);
		if (n_chars > SUMMARY_LINE_SIZE) {
			// Trim name
		}
		int n_spaces = SUMMARY_LINE_SIZE - n_chars - 2; // Count ":" and one space too
		get_format (fmt, i);
		fprintf (fp, "%s: ", scenario_expr_formulas[i].name);
		for (int i = 0; i < n_spaces; i++) {
			fputc (' ', fp);
		}
                fprintf (fp, fmt, scenario_expr_formulas[i].value);
		fprintf (fp, "%s\n", scenario_expr_format[i].unit);
	}
}

/**********************************************************************/

void scenario_expr_print_column (FILE *fp, int i_scenario) {
	fprintf (fp, formats[i_scenario], scenario_expr_formulas[i_scenario].value);
}

/**********************************************************************/

void scenario_expr_print_all_columns (FILE *fp) {
	for (int i = 0; i < scenario_expr_n_formulas; i++) {
		scenario_expr_print_column (fp, i);
	}
}

/**********************************************************************/

void scenario_expr_print_raw_counters (FILE *f) {
	for (int i = 0; i < scenario_expr_n_vars; i++) {
		fprintf (f, "%-37s : %20ld\n", scenario_expr_counter_names[i],
			 (long) scenario_expr_counter_values[i]);
	}
}

/**********************************************************************/

double scenario_expr_get_value (int i_scenario) {
	return scenario_expr_formulas[i_scenario].value;
}

/**********************************************************************/

int scenario_expr_get_column_width (int i_scenario) {
	return scenario_expr_format[i_scenario].decpl_1 + 1;
}

/**********************************************************************/

int scenario_expr_get_table_width () {
	int tw = 0;
	for (int i = 0; i < scenario_expr_n_formulas; i++) {
		if (scenario_expr_format[i].decpl_1 == 0) {
			tw += n_decimal_places(i) + scenario_expr_format[i].decpl_2 + 1;
		} else {
			tw += scenario_expr_format[i].decpl_1 + 1;
		}
	}
	return tw;
}

/**********************************************************************/

void scenario_expr_unique_group_indices (int *n_groups, int *is_unique, int id) {
	char *tmp[scenario_expr_n_formulas];
	for (int i = 0; i < scenario_expr_n_formulas; i++) {
		tmp[i] = "";
	}
	*n_groups = 0;
	for (int i = 0; i < scenario_expr_n_formulas; i++) {
		int found = 0;
		for (int j = 0; j < *n_groups; j++) {
			if (id == 0) {
				found = found || !strcmp (tmp[j], scenario_expr_format[i].group);
			} else if (id == 1) {
				found = found || !strcmp (tmp[j], scenario_expr_format[i].column1);
			}
		}
		if (!found) {
			if (id == 0) {
				tmp[*n_groups] = strdup (scenario_expr_format[i].group);
			} else if (id == 1) {
				tmp[*n_groups] = strdup (scenario_expr_format[i].column1);
			}
			is_unique[i] = 1;
			*n_groups = *n_groups + 1;
		} else {
			is_unique[i] = 0;
		}
	}
}

/**********************************************************************/

void print_formatted (FILE *fp, int *is_unique, int id) {
	char all_columns[80] = "";
	int select;
	for (int i = 0; i < scenario_expr_n_formulas; i++) {
		if (is_unique == NULL) {
			select = 0;
		} else {
			select = is_unique[i];
		}
		if (select) {
			if (id == 0) {
				strcat (all_columns, strdup(scenario_expr_format[i].group));
			} else if (id == 1) {
				strcat (all_columns, strdup(scenario_expr_format[i].column1));
			} else if (id == 2) {
				strcat (all_columns, strdup(scenario_expr_format[i].column2));
			}
			strcat (all_columns, " ");
		}
	}
	char *foo = all_columns;
	int tw = scenario_expr_get_table_width ();
	int ns = strlen (foo);
	int nc = ns < tw ? ns : tw;
	int nb = ns < tw ? tw - ns + 1 : 1;
	for (int is = 0; is < nc; is++) {
		fputc (*foo++, fp);
	}
	for (int is = 0; is < nb; is++) {
		fputc (' ', fp);
	}	
}

/**********************************************************************/

void scenario_expr_print_group (FILE *fp) {
	int n_groups;
	int is_unique[scenario_expr_n_formulas];
	scenario_expr_unique_group_indices (&n_groups, is_unique, 0);
	print_formatted (fp, is_unique, 0);
}

/**********************************************************************/

void scenario_expr_print_subgroup (FILE *fp) {
	int n_groups;
	int is_unique[scenario_expr_n_formulas];
	scenario_expr_unique_group_indices (&n_groups, is_unique, 1);
	print_formatted (fp, is_unique, 1);
}

/**********************************************************************/

void scenario_expr_print_header (FILE *fp) {
	char all_columns[80] = "";
	for (int i = 0; i < scenario_expr_n_formulas; i++) {
		strcat (all_columns, strdup(scenario_expr_format[i].column2));
		strcat (all_columns, " ");
	}
	char *foo = all_columns;
	int tw = scenario_expr_get_table_width ();
	int ns = strlen (foo);
	int nc = ns < tw ? ns : tw;
	int nb = ns < tw ? tw - ns + 1 : 1;
	for (int is = 0; is < nc; is++) {
		fputc (*foo++, fp);
	}
	for (int is = 0; is < nb; is++) {
		fputc (' ', fp);
	}	
}

/**********************************************************************/

void scenario_expr_add_veperf_counters () {
	for (int i = 0; i < scenario_expr_n_vars; i++) {
		vftr_sx_counter (scenario_expr_counter_names[i], i);	
	}
}

/**********************************************************************/

void scenario_expr_add_papi_counters () {
	for (int i = 0; i < scenario_expr_n_vars; i++) {
		vftr_papi_counter (scenario_expr_counter_names[i]);
	}
}

/**********************************************************************/

int vftr_scenario_test_1 (FILE *fp_in, FILE *fp_out) {
	vftr_read_scenario_file ("", fp_in);
	fprintf (fp_out, "Registered variables: %d\n", scenario_expr_n_vars);
	for (int i = 0; i < scenario_expr_n_vars; i++) {
		fprintf (fp_out, "%d: name: %s\n", i, te_vars[i].name);		
	}
	fprintf (fp_out, "Check for the three additional entries: \n");
	fprintf (fp_out, "%s\n", te_vars[scenario_expr_n_vars].name);
	fprintf (fp_out, "%s\n", te_vars[scenario_expr_n_vars+1].name);
	fprintf (fp_out, "%s\n", te_vars[scenario_expr_n_vars+2].name);
	fprintf (fp_out, "Registered formulas: %d\n", scenario_expr_n_formulas);
	for (int i = 0; i < scenario_expr_n_formulas; i++) {
		vftr_scenario_print_formula (fp_out, scenario_expr_formulas[i]);
	}
}

/**********************************************************************/

int vftr_scenario_test_2 (FILE *fp_in, FILE *fp_out) {
	vftr_read_scenario_file ("", fp_in);
	scenario_expr_counter_values[0] = 1.5; // c1
	scenario_expr_counter_values[1] = 0.5; // c2
	scenario_expr_counter_values[2] = -1.0; // c3
	scenario_expr_evaluate_all (0.0, 0ll);
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
	fprintf (fp_out, "Check standard operations: \n");
	fprintf (fp_out, "%s: %lf\n", scenario_expr_formulas[0].name, scenario_expr_formulas[0].value);
	fprintf (fp_out, "%s: %lf\n", scenario_expr_formulas[1].name, scenario_expr_formulas[1].value);
	fprintf (fp_out, "%s: %lf\n", scenario_expr_formulas[2].name, scenario_expr_formulas[2].value);
	fprintf (fp_out, "%s: %lf\n", scenario_expr_formulas[3].name, scenario_expr_formulas[3].value);
	fprintf (fp_out, "%s: %lf\n", scenario_expr_formulas[4].name, scenario_expr_formulas[4].value);
	fprintf (fp_out, "%s: %lf\n", scenario_expr_formulas[5].name, scenario_expr_formulas[5].value);
	fprintf (fp_out, "%s: %lf\n", scenario_expr_formulas[6].name, scenario_expr_formulas[6].value);
	fprintf (fp_out, "%s: %lf\n", scenario_expr_formulas[7].name, scenario_expr_formulas[7].value);
	fprintf (fp_out, "%s: %lf\n", scenario_expr_formulas[8].name, scenario_expr_formulas[8].value);
	
	fprintf (fp_out, "Check that division by zero is protected: \n");
	scenario_expr_counter_values[1] = 0.0;
	scenario_expr_evaluate (3, 0.0, 0ll);
	fprintf (fp_out, "%s: %lf\n", scenario_expr_formulas[3].name, scenario_expr_formulas[3].value);
// Check that division by zero is protected
}

/**********************************************************************/

// This test expects a scenario file with wrong syntax in the formulas .
// It should be tagged as XFAIL.

int vftr_scenario_test_3 (FILE *fp_in, FILE *fp_out) {
	vftr_read_scenario_file ("", fp_in);
}

/**********************************************************************/
