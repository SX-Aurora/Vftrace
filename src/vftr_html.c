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

#include <string.h>

#include "vftr_stacks.h"

/**********************************************************************/

#define INDENT_SPACES 3

void vftr_make_html_indent (FILE *fp, int n_indent_0, int n_indent_extra) {
	for (int i = 0; i < n_indent_0 + n_indent_extra * INDENT_SPACES; i++) fprintf (fp, " ");
}

void vftr_print_html_function_element (FILE *fp, int stack_id, int func_id, int n_spaces) {
	vftr_make_html_indent (fp, n_spaces, 0);
	fprintf (fp, "<a hfref=\"#\">%s\n", vftr_gStackinfo[stack_id].name);
	vftr_make_html_indent (fp, n_spaces + 3, 0);
	fprintf (fp, "<ttt class=\"ttt\">function name: %s\n", vftr_gStackinfo[stack_id].name);
	vftr_make_html_indent (fp, n_spaces + 3, 0);
	if (func_id < 0) {
	   fprintf (fp, "<br>n_calls: %d\n", -1);
	} else {
	   fprintf (fp, "<br>n_calls: %d\n", vftr_func_table[func_id]->prof_current.calls);
	}
	vftr_make_html_indent (fp, n_spaces + 3, 0);
	fprintf (fp, "</ttt>\n");
	vftr_make_html_indent (fp, n_spaces, 0);
	fprintf (fp, "</a>\n");
}

/**********************************************************************/

vftr_print_css_header (FILE *fp) {

   fprintf (fp, "<style>\n");
   fprintf (fp, "*,\n");
   fprintf (fp, "*:before,\n");
   fprintf (fp, "*:after {\n");
   fprintf (fp, "  -webkit-box-sizing: border-box;\n");
   fprintf (fp, "  -moz-box-sizing: border-box;\n");
   fprintf (fp, "  box-sizing: border-box;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, "* {\n");
   fprintf (fp, "  position: relative;\n");
   fprintf (fp, "  margin: 0;\n");
   fprintf (fp, "  padding: 0;\n");
   fprintf (fp, "\n"); 
   fprintf (fp, "  border: 0 none;\n");
   fprintf (fp, "\n"); 
   fprintf (fp, "  -webkit-transition: all ease .4s;\n");
   fprintf (fp, "  -moz-transition: all ease .4s;\n");
   fprintf (fp, "    transition: all ease .4s;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, "html, body {\n");
   fprintf (fp, "  width: 100%;\n");
   fprintf (fp, "  height: 100%;\n");
   fprintf (fp, "  background: white;\n");
   fprintf (fp, "  font-family: 'Fjalla One', sans-serif;\n");
   fprintf (fp, "  font-size: 18px;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, "h1 {\n");
   fprintf (fp, "  padding-top: 40px;\n");
   fprintf (fp, "  color: black;\n");
   fprintf (fp, "  text-align: center;\n");
   fprintf (fp, "  font-size: 1.8rem;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav {\n");
   fprintf (fp, "  margin: 100px auto;\n");
   fprintf (fp, "  width: 5000px;\n");
   fprintf (fp, "  min-height: auto;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav ul {\n");
   fprintf (fp, "  position: relative;\n");
   fprintf (fp, "  padding-top: 20px;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li {\n");
   fprintf (fp, "  position: relative;\n");
   fprintf (fp, "  padding: 20px 3px 0 3px;\n");
   fprintf (fp, "  float: left;\n");
   fprintf (fp, "\n");
   fprintf (fp, "  text-align: center;\n");
   fprintf (fp, "  list-style-type: none;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li::before, .nav li::after{\n");
   fprintf (fp, "  content: '';\n");
   fprintf (fp, "  position: absolute;\n");
   fprintf (fp, "  top: 0;\n");
   fprintf (fp, "  right: 50%;\n");
   fprintf (fp, "  width: 50%;\n");
   fprintf (fp, "  height: 20px;\n");
   fprintf (fp, "  border-top: 1px solid black;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li::after{\n");
   fprintf (fp, "  left: 50%;\n");
   fprintf (fp, "  right: auto;\n");
   fprintf (fp, "\n");  
   fprintf (fp, "  border-left: 1px solid black;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li:only-child::after, .nav li:only-child::before {\n");
   fprintf (fp, "  content: '';\n");
   fprintf (fp, "  display: none;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li:only-child{ padding-top: 0;}\n");
   fprintf (fp, ".nav li:first-child::before, .nav li:last-child::after{\n");
   fprintf (fp, "  border: 0 none;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li:last-child::before{\n");
   fprintf (fp, "  border-right: 1px solid black;\n");
   fprintf (fp, "  border-radius: 0 5px 0 0;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li:first-child::after{\n");
   fprintf (fp, "    border-radius: 5px 0 0 0;\n");
   fprintf (fp, "}\n");
   fprintf (fp, ".nav ul ul::before{\n");
   fprintf (fp, "  content: '';\n");
   fprintf (fp, "  position: absolute; top: 0; left: 50%;\n");
   fprintf (fp, "  border-left: 1px solid black;\n");
   fprintf (fp, "  width: 0;\n");
   fprintf (fp, "  height: 20px;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li a{\n");
   fprintf (fp, "  display: inline-block;\n");
   fprintf (fp, "  padding: 5px 10px;\n");
   fprintf (fp, "  border-radius: 5px;\n");
   fprintf (fp, "  border: 1px solid black;\n");
   fprintf (fp, "  text-decoration: none;\n");
   fprintf (fp, "  text-transform: uppercase;\n");
   fprintf (fp, "  background-color: #edebeb;\n");
   fprintf (fp, "  color: black;\n");
   fprintf (fp, "  font-family: arial, verdana, tahoma;\n");
   fprintf (fp, "  font-size: 11px;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, "/* This is the stack info box which appears when hovering over a function node */\n");
   fprintf (fp, "\n");
   fprintf (fp, "a .ttt {\n");
   fprintf (fp, "  visibility: hidden;\n");
   fprintf (fp, "  width: 100%;\n");
   fprintf (fp, "  background-color: black;\n");
   fprintf (fp, "  color: #fff;\n");
   fprintf (fp, "  text-align: center;\n");
   fprintf (fp, "  border-radius: 6px; /* Rounded corners */\n");
   fprintf (fp, "  padding: 10px 0;\n");
   fprintf (fp, "  position: absolute;\n");
   fprintf (fp, "  z-index: 1; /* Precedence of this element - atop of the leaf. */\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, "a:hover .ttt{\n");
   fprintf (fp, "  visibility: visible;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, "hr {\n");
   fprintf (fp, "  display: block;\n");
   fprintf (fp, "  height: 1px;\n");
   fprintf (fp, "  border: 0;\n");
   fprintf (fp, "  border-top: 1px solid black;\n");
   fprintf (fp, "  margin: 1em 0;\n");
   fprintf (fp, "  padding: 0;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "}\n");
   fprintf (fp, "</style>\n");
}

/**********************************************************************/

void vftr_print_stacktree_to_html (FILE *fp, stack_leaf_t *leaf, int n_spaces, long long *total_time, double *imbalances) {
	if (!leaf) return;
	vftr_print_html_function_element (fp, leaf->stack_id, leaf->func_id, n_spaces);
	if (leaf->callee) {
		vftr_make_html_indent (fp, n_spaces, 0);
		fprintf (fp, "<ul>\n");
		vftr_make_html_indent (fp, n_spaces, 1);
		fprintf (fp, "<li>\n");
		vftr_print_stacktree_to_html (fp, leaf->callee, n_spaces + 2 * INDENT_SPACES, 0, imbalances);
		vftr_make_html_indent (fp, n_spaces, 1);
		fprintf (fp, "</li>\n");
		vftr_make_html_indent (fp, n_spaces, 0);
		fprintf (fp, "</ul>\n");
	} else {
	       vftr_make_html_indent (fp, n_spaces, 0);
	       fprintf (fp, "<hr/>\n");
	       if (leaf->func_id < 0) { 
	          vftr_make_html_indent (fp, n_spaces, 0);
		  fprintf (fp, "[not on this rank]\n");
	       } else { 
	          vftr_make_html_indent (fp, n_spaces, 0);
	          fprintf (fp, "Position: %d<br>\n", leaf->final_id);
	          vftr_make_html_indent (fp, n_spaces, 0);
		  fprintf (fp, "Time %lf s<br>\n", vftr_func_table[leaf->func_id]->prof_current.timeIncl * 1e-6);
	          vftr_make_html_indent (fp, n_spaces, 0);
		  if (imbalances != NULL) fprintf (fp, "Imbalance: %lf %<br>\n", imbalances[leaf->func_id]);
		  double n_bytes = vftr_func_table[leaf->func_id]->prof_current.mpi_tot_send_bytes;
		  char *unit_str;
		  vftr_memory_unit (&n_bytes, &unit_str);
	          vftr_make_html_indent (fp, n_spaces, 0);
		  fprintf (fp, "Send: %lf %s<br>\n", n_bytes, unit_str);
		  n_bytes = vftr_func_table[leaf->func_id]->prof_current.mpi_tot_recv_bytes;
		  vftr_memory_unit (&n_bytes, &unit_str);
	          vftr_make_html_indent (fp, n_spaces, 0);
		  fprintf (fp, "Recv: %lf %s<br>\n", n_bytes, unit_str);
	     }
	}
		

	if (leaf->next_in_level) {
		vftr_make_html_indent (fp, n_spaces, 0);
		fprintf (fp, "<li>\n");
		vftr_print_stacktree_to_html (fp, leaf->next_in_level, n_spaces + INDENT_SPACES, 0, imbalances);
		vftr_make_html_indent (fp, n_spaces, 0);
		fprintf (fp, "</li>\n");
	}
}

/**********************************************************************/

void vftr_print_html_output (FILE *fp_out, char *func_name, stack_leaf_t *leaf, double *imbalances) {
	// No external file (e.g. for tests) given. Create filename <func_name>.html
	FILE *fp;
	if (!fp_out) {
	   char html_filename[strlen(func_name) + 6];
	   snprintf (html_filename, strlen(func_name) + 6, "%s.html", func_name);
	   fp = fopen (html_filename, "w");
        } else {
	   fp = fp_out;
        }
	vftr_print_css_header (fp);
	fprintf (fp, "<h1>Vftrace stack tree for %s</h1>\n", func_name);
	fprintf (fp, "<link rel=\"stylesheet\" href=\"/usr/uhome/aurora/ess/esscw/tmp/flow.css\">\n");
	fprintf (fp, "<nav class=\"nav\"/>\n");
	vftr_make_html_indent (fp, 0, 1);
	fprintf (fp, "<ul>\n");
	vftr_make_html_indent (fp, 0, 2);
	fprintf (fp, "<li>\n");
	vftr_print_stacktree_to_html (fp, leaf->origin, 3 * INDENT_SPACES, 0, imbalances);
	vftr_make_html_indent (fp, 0, 2);
	fprintf (fp, "</li>\n");
	vftr_make_html_indent (fp, 0, 1);
	fprintf (fp, "</ul>\n");
	fprintf (fp, "</nav>\n");
	if (!fp_out) fclose (fp); //External file is supposed to be closed elsewhere
}

/**********************************************************************/

int vftr_html_test_1 (FILE *fp_in, FILE *fp_out) {
	unsigned long long addrs[6];	
	function_t *func1 = vftr_new_function (NULL, "init", NULL, 0, false);
	function_t *func2 = vftr_new_function ((void*)addrs, "MAIN__", func1, 0, false);
	function_t *func3 = vftr_new_function ((void*)(addrs + 1), "A", func1, 0, false);
	function_t *func4 = vftr_new_function ((void*)(addrs + 2), "B", func1, 0, false);
	function_t *func5 = vftr_new_function ((void*)(addrs + 3), "C", func3, 0, false);
	function_t *func6 = vftr_new_function ((void*)(addrs + 4), "C", func4, 0, false);
	vftr_normalize_stacks();
	for (int i = 0; i < vftr_stackscount; i++) {
		vftr_func_table[i]->prof_current.calls = i + 1;
	}

	int *stack_indices, *func_indices;
	int n_indices;
	vftr_find_function_in_table ("C", &func_indices, &n_indices, false);	
	vftr_find_function_in_stack ("C", &stack_indices, &n_indices, false);
 	stack_leaf_t *stack_tree = NULL;
	for (int i = 0;  i < n_indices; i++) {
		//int n_functions_in_stack = vftr_stack_length (stack_indices[i]);
		// Why is n_functions_in_stack 2 instead of 3?
		int n_functions_in_stack = 3;
		int *stack_ids = (int*)malloc (n_functions_in_stack * sizeof(int));	
		int stack_id = stack_indices[i];
		int function_id = func_indices[i];
		for (int j = 0; j < n_functions_in_stack; j++) {
			stack_ids[j] = stack_id;
			stack_id = vftr_gStackinfo[stack_id].ret;
			printf ("%d ", stack_id);
		}
		printf ("\n");
		vftr_fill_into_stack_tree (&stack_tree, n_functions_in_stack, stack_ids, function_id);
		free (stack_ids);
	}
	long long dummy1;
	double dummy2;
	vftr_print_html_output (fp_out, "C", stack_tree->origin, NULL);
	free (stack_tree);
	return 0;
}

/**********************************************************************/
