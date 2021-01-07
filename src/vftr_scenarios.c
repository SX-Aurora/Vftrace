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
#include <stdbool.h>

#include "vftr_signals.h"
#include "vftr_fileutils.h"
#include "vftr_functions.h"
#include "vftr_hwcounters.h"
#include "vftr_scenarios.h"
#include "jsmn.h"
#include "tinyexpr.h"

char *vftr_scenario_expr_counter_names[TE_MAX];
char *vftr_scenario_expr_vars[TE_MAX];
function_expr_t vftr_scenario_expr_formulas[TE_MAX];
hwc_format_t vftr_scenario_expr_format[TE_MAX];

int vftr_scenario_expr_n_vars;
int vftr_scenario_expr_n_formulas;

double *vftr_scenario_expr_counter_values;
double vftr_scenario_expr_runtime;
double vftr_scenario_expr_cycles;
double vftr_scenario_expr_cycletime;

/**********************************************************************/

void vftr_init_scenario_formats () {
   for (int i = 0; i < TE_MAX; i++) {
      vftr_scenario_expr_format[i].header = NULL;
      vftr_scenario_expr_format[i].unit = NULL; 
      vftr_scenario_expr_format[i].decimal_places = 0;
   }
}

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
	fwrite (&vftr_scenario_expr_n_formulas, sizeof(int), 1, fp);
#if defined(HAS_SXHWC) || defined(HAS_PAPI)
	for (int i = 0; i < vftr_scenario_expr_n_formulas; i++) {
		fwrite (vftr_scenario_expr_formulas[i].name, SCENARIO_NAME_LEN, 1, fp);
		fwrite (&vftr_scenario_expr_formulas[i].integrated, sizeof(int), 1, fp);
	}
#endif
}

/**********************************************************************/

void vftr_write_observables_to_vfd (profdata_t *prof_current, profdata_t *prof_previous, FILE *fp) {
#if defined(HAS_SXHWC) || defined(HAS_PAPI)
        for (int i = 0; i < vftr_n_hw_obs; i++) {
		double value;
		if (prof_current != NULL && prof_previous != NULL) {
                  value = (double)(prof_current->event_count[i] - prof_previous->event_count[i]);
		} else { // Dummy entry, e.g. for vftr_finalize
		  value = 0.0;
                }
		//fwrite (&vftr_scenario_expr_formulas[i].value, sizeof(double), 1, fp);
		fwrite (&value, sizeof(double), 1, fp);
	}
#endif
}

/**********************************************************************/

void vftr_trim_trailing_white_spaces (char *s) {
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

bool read_counters;
bool read_observables;
bool read_hwc_name;
bool read_name;
bool read_symbol;
bool read_formula;
bool read_runtime;
bool read_protected;
bool read_default;
bool read_unit;
bool read_header;
bool read_decimal_places;

int vftr_read_json (const char *js, jsmntok_t *tok, size_t count) {
	jsmntok_t *key;
	if (tok->type == JSMN_PRIMITIVE) {
		return 1;
	} else if (tok->type == JSMN_STRING) {
		char *s = strndup(js + tok->start, tok->end - tok->start);
		if (read_hwc_name) {
			vftr_scenario_expr_counter_names[vftr_scenario_expr_n_vars] = strdup(s);
			read_hwc_name = false;
		} else if (read_counters && read_symbol) {
			vftr_scenario_expr_vars[vftr_scenario_expr_n_vars] = strdup(s);
			read_symbol = false;
			vftr_scenario_expr_n_vars++;
		} else if (read_name) {
			vftr_scenario_expr_formulas[vftr_scenario_expr_n_formulas++].name = strdup(s);
			vftr_scenario_expr_formulas[vftr_scenario_expr_n_formulas-1].protected_values = NULL;
			read_name = false;
		} else if (read_formula) {
			vftr_scenario_expr_formulas[vftr_scenario_expr_n_formulas-1].formula = strdup(s);		
			read_formula = false;
		} else if (read_runtime) {
			if (!strcmp (s, "yes")) {
				vftr_scenario_expr_formulas[vftr_scenario_expr_n_formulas-1].integrated = false;
			}
			read_runtime = false;	
		} else if (read_protected) {
			vftr_scenario_expr_formulas[vftr_scenario_expr_n_formulas-1].protected_values = strdup(s);
			read_protected = false;
		} else if (read_default) {
			sscanf (s, "%lf", &vftr_scenario_expr_formulas[vftr_scenario_expr_n_formulas-1].default_value);	
			read_default = false;
		} else if (read_unit) {
			if (strcmp (s, "")) vftr_scenario_expr_format[vftr_scenario_expr_n_formulas-1].unit = strdup (s);
			read_unit = false;
		} else if (read_header) {
			vftr_scenario_expr_format[vftr_scenario_expr_n_formulas-1].header = strdup(s);
			read_header = false;
		} else if (read_decimal_places) {
			vftr_scenario_expr_format[vftr_scenario_expr_n_formulas-1].decimal_places = atoi(s);
			read_decimal_places = false;
		}

		if (!strcmp (s, "counters")) {
			read_counters = true;
			read_observables = false;
		} else if (!strcmp (s, "observables")) {
			read_observables = true;
			read_counters = false;
		} else if (!strcmp (s, "hwc_name")) {
			read_hwc_name = true;
		} else if (!strcmp (s, "name")) {
			read_name = true;
		} else if (!strcmp (s, "symbol")) {
			read_symbol = true;
		} else if (!strcmp (s, "formula")) {
			read_formula = true;
		} else if (!strcmp (s, "divide_by_runtime")) {
			read_runtime = true;
		} else if (!strcmp (s, "protected")) {
			read_protected = true;
		} else if (!strcmp (s, "default")) {
			read_default = true;
		} else if (!strcmp (s, "unit")) {
			read_unit = true;
		} else if (!strcmp (s, "decimal_places")) {
			read_decimal_places = true;
		} else if (!strcmp (s, "header")) {
			read_header = true;
		}
		return 1;
	} else if (tok->type == JSMN_OBJECT) {
		int j = 0;
		for (int i = 0; i < tok->size; i++) {
			key = tok + j + 1;
			j += vftr_read_json (js, key, count - j);
			if (key->size > 0) {
				j += vftr_read_json (js, tok + j + 1, count - j);
			}
		}
		return j + 1;
	} else if (tok->type == JSMN_ARRAY) {
		int j = 0;
		for (int i = 0; i < tok->size; i++) {
			j += vftr_read_json (js, tok + j + 1, count - j);
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
		vftr_scenario_expr_formulas[i].integrated = true;
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
			vftr_read_json (js, token, token_count);	
		}
	}
	if (need_to_close_file) fclose (fp);
	
	te_vars = (te_variable *) malloc ((vftr_scenario_expr_n_vars + 3) * sizeof (te_variable));
	vftr_scenario_expr_counter_values = (double *) malloc (vftr_scenario_expr_n_vars * sizeof (double));
	for (int i = 0; i < vftr_scenario_expr_n_vars; i++) {
		te_vars[i].name = vftr_scenario_expr_vars[i];
		te_vars[i].address = &vftr_scenario_expr_counter_values[i];
		te_vars[i].type = 0;
		te_vars[i].context = NULL;
	}	
	te_vars[vftr_scenario_expr_n_vars].name = "runtime";
	te_vars[vftr_scenario_expr_n_vars].address = &vftr_scenario_expr_runtime;
	te_vars[vftr_scenario_expr_n_vars].type = 0;
	te_vars[vftr_scenario_expr_n_vars].context = NULL;

	te_vars[vftr_scenario_expr_n_vars+1].name = "cycles";
	te_vars[vftr_scenario_expr_n_vars+1].address = &vftr_scenario_expr_cycles;
	te_vars[vftr_scenario_expr_n_vars+1].type = 0;
	te_vars[vftr_scenario_expr_n_vars+1].context = NULL;

	te_vars[vftr_scenario_expr_n_vars+2].name = "cycletime";
	te_vars[vftr_scenario_expr_n_vars+2].address = &vftr_scenario_expr_cycletime;
	te_vars[vftr_scenario_expr_n_vars+2].type = 0;
	te_vars[vftr_scenario_expr_n_vars+2].context = NULL;

	int err;
	expr = (te_expr **) malloc (vftr_scenario_expr_n_formulas * sizeof (te_expr *));	
	for (int i = 0; i < vftr_scenario_expr_n_formulas; i++) {
		expr[i] = te_compile (vftr_scenario_expr_formulas[i].formula, te_vars, vftr_scenario_expr_n_vars + 3, &err);
		if (!expr[i]) {
			printf ("ERROR COMPILING FORMULA:\n%s\n", vftr_scenario_expr_formulas[i].formula);
			printf ("%*s^\n", err - 1, "");
			vftr_abort();
		}
			
	}
	return 0;
}

/**********************************************************************/

int vftr_scenario_variable_index (char *varname) {
	for (int i = 0; i < vftr_scenario_expr_n_vars + 1; i++) {
		if (!strcmp (varname, te_vars[i].name)) {
			return i;
		}
	}
	return -1;
}

/**********************************************************************/

void vftr_scenario_expr_evaluate (int i_scenario, double runtime, unsigned long long cycles) {
	vftr_scenario_expr_runtime = runtime;
	vftr_scenario_expr_cycles = (double)cycles;
	vftr_scenario_expr_cycletime = runtime / cycles;

	int i_protected = vftr_scenario_expr_formulas[i_scenario].protected_values ?
		vftr_scenario_variable_index (vftr_scenario_expr_formulas[i_scenario].protected_values) : -1;
	double check_value = -1.0;
	if (i_protected < 0) {
		vftr_scenario_expr_formulas[i_scenario].value = te_eval (expr[i_scenario]);		
	} else if (i_protected == vftr_scenario_expr_n_vars) {
		check_value = vftr_scenario_expr_runtime;
	} else if (i_protected == vftr_scenario_expr_n_vars + 1) {
		check_value = vftr_scenario_expr_runtime;
	} else {
		check_value = (double)vftr_scenario_expr_counter_values[i_protected];
	}
	if (check_value == 0.) {
		vftr_scenario_expr_formulas[i_scenario].value = vftr_scenario_expr_formulas[i_scenario].default_value;	
	} else {
		vftr_scenario_expr_formulas[i_scenario].value = te_eval (expr[i_scenario]);
		if (!vftr_scenario_expr_formulas[i_scenario].integrated && runtime > 0.) {
			vftr_scenario_expr_formulas[i_scenario].value /= runtime;
		}
	}
}

/**********************************************************************/

void vftr_scenario_expr_evaluate_all (double runtime, unsigned long long cycles) {
	for (int i = 0; i < vftr_scenario_expr_n_formulas; i++) {
		vftr_scenario_expr_evaluate (i, runtime, cycles);
	}
}

/**********************************************************************/
		
static void vftr_scenario_get_format (char *fmt, int i) {
	int behind_comma = vftr_scenario_expr_format[i].decimal_places;
	static int total = 0;
	int tmp = vftr_count_digits_int(i);
	total = tmp > total ? tmp : total;
	sprintf (fmt, "%%%d.%dlf ", total, behind_comma);
}

/**********************************************************************/

#define SUMMARY_LINE_SIZE 27
void vftr_scenario_expr_print_summary (FILE *fp) {
	char fmt[10];
	for (int i = 0; i < vftr_scenario_expr_n_formulas; i++) {
		int n_chars = strlen(vftr_scenario_expr_formulas[i].name);
		if (n_chars > SUMMARY_LINE_SIZE) {
			// Trim name
		}
		int n_spaces = SUMMARY_LINE_SIZE - n_chars - 2; // Count ":" and one space too
		vftr_scenario_get_format (fmt, i);
		fprintf (fp, "%s: ", vftr_scenario_expr_formulas[i].name);
		for (int i = 0; i < n_spaces; i++) {
			fputc (' ', fp);
		}
                fprintf (fp, fmt, vftr_scenario_expr_formulas[i].value);
		fprintf (fp, "%s\n", vftr_scenario_expr_format[i].unit);
	}
}

/**********************************************************************/

void vftr_scenario_expr_print_raw_counters (FILE *f) {
	for (int i = 0; i < vftr_scenario_expr_n_vars; i++) {
		fprintf (f, "%-37s : %20ld\n", vftr_scenario_expr_counter_names[i],
			 (long) vftr_scenario_expr_counter_values[i]);
	}
}

/**********************************************************************/

double vftr_scenario_expr_get_value (int i_scenario) {
	return vftr_scenario_expr_formulas[i_scenario].value;
}

/**********************************************************************/

void vftr_scenario_expr_add_sx_counters () {
	for (int i = 0; i < vftr_scenario_expr_n_vars; i++) {
		vftr_sx_counter (vftr_scenario_expr_counter_names[i], i);	
	}
}

/**********************************************************************/

void vftr_scenario_expr_add_papi_counters () {
	for (int i = 0; i < vftr_scenario_expr_n_vars; i++) {
		vftr_papi_counter (vftr_scenario_expr_counter_names[i]);
	}
}

/**********************************************************************/

int vftr_scenario_test_1 (FILE *fp_in, FILE *fp_out) {
	vftr_read_scenario_file ("", fp_in);
	fprintf (fp_out, "Registered variables: %d\n", vftr_scenario_expr_n_vars);
	for (int i = 0; i < vftr_scenario_expr_n_vars; i++) {
		fprintf (fp_out, "%d: name: %s\n", i, te_vars[i].name);		
	}
	fprintf (fp_out, "Check for the three additional entries: \n");
	fprintf (fp_out, "%s\n", te_vars[vftr_scenario_expr_n_vars].name);
	fprintf (fp_out, "%s\n", te_vars[vftr_scenario_expr_n_vars+1].name);
	fprintf (fp_out, "%s\n", te_vars[vftr_scenario_expr_n_vars+2].name);
	fprintf (fp_out, "Registered formulas: %d\n", vftr_scenario_expr_n_formulas);
	for (int i = 0; i < vftr_scenario_expr_n_formulas; i++) {
		vftr_scenario_print_formula (fp_out, vftr_scenario_expr_formulas[i]);
	}
	return 0;
}

/**********************************************************************/

int vftr_scenario_test_2 (FILE *fp_in, FILE *fp_out) {
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
	fprintf (fp_out, "Check standard operations: \n");
	fprintf (fp_out, "%s: %lf\n", vftr_scenario_expr_formulas[0].name, vftr_scenario_expr_formulas[0].value);
	fprintf (fp_out, "%s: %lf\n", vftr_scenario_expr_formulas[1].name, vftr_scenario_expr_formulas[1].value);
	fprintf (fp_out, "%s: %lf\n", vftr_scenario_expr_formulas[2].name, vftr_scenario_expr_formulas[2].value);
	fprintf (fp_out, "%s: %lf\n", vftr_scenario_expr_formulas[3].name, vftr_scenario_expr_formulas[3].value);
	fprintf (fp_out, "%s: %lf\n", vftr_scenario_expr_formulas[4].name, vftr_scenario_expr_formulas[4].value);
	fprintf (fp_out, "%s: %lf\n", vftr_scenario_expr_formulas[5].name, vftr_scenario_expr_formulas[5].value);
	fprintf (fp_out, "%s: %lf\n", vftr_scenario_expr_formulas[6].name, vftr_scenario_expr_formulas[6].value);
	fprintf (fp_out, "%s: %lf\n", vftr_scenario_expr_formulas[7].name, vftr_scenario_expr_formulas[7].value);
	fprintf (fp_out, "%s: %lf\n", vftr_scenario_expr_formulas[8].name, vftr_scenario_expr_formulas[8].value);
	
	fprintf (fp_out, "Check that division by zero is protected: \n");
	vftr_scenario_expr_counter_values[1] = 0.0;
	vftr_scenario_expr_evaluate (3, 0.0, 0ll);
	fprintf (fp_out, "%s: %lf\n", vftr_scenario_expr_formulas[3].name, vftr_scenario_expr_formulas[3].value);
	return 0;
// Check that division by zero is protected
}

/**********************************************************************/

// This test expects a scenario file with wrong syntax in the formulas .
// It should be tagged as XFAIL.

int vftr_scenario_test_3 (FILE *fp_in, FILE *fp_out) {
	vftr_read_scenario_file ("", fp_in);
	return 0;
}

/**********************************************************************/
