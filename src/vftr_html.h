#ifndef VFTR_HTML_H
#define VFTR_HTML_H

#include "vftr_filewrite.h"
#include "vftr_stacks.h"

enum origin_page {HOME, TREE, PROFILE};

void vftr_make_html_indent (FILE *fp, int n_indent_0, int n_indent_extra);
void vftr_print_css_header (FILE *fp);
void vftr_print_html_output (FILE *fp_out, char *func_names[], int n_funcs, int this_i_func,
			     stack_leaf_t *leaf, double *imbalances, double total_time);
void vftr_print_navigation_bars (FILE *fp, char *func_names[], int n_funcs, int this_i_func, enum origin_page op);
void vftr_print_index_html (char *func_names[], int n_funcs);

int vftr_html_test_1 (FILE *fp_in, FILE *fp_out);
#endif
