#ifndef VFTR_HTML_H
#define VFTR_HTML_H

#include "vftr_filewrite.h"
#include "vftr_stacks.h"

void vftr_make_html_indent (FILE *fp, int n_indent_0, int n_indent_extra);
void vftr_print_html_output (FILE *fp_out, display_function_t **display_funcs, int n_display_functions, int  this_i_func,
			     stack_leaf_t *leaf, double *imbalances, double total_time);

int vftr_html_test_1 (FILE *fp_in, FILE *fp_out);
#endif
