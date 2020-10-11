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

void vftr_print_stacktree_to_html (FILE *fp, stack_leaf_t *leaf, int n_spaces, long long *total_time) {
	if (!leaf) return;
	vftr_make_html_indent (fp, n_spaces, 0);
	fprintf (fp, "<a hfref=\"#\">%s</a>\n", vftr_gStackinfo[leaf->stack_id].name);
	if (leaf->callee) {
		vftr_make_html_indent (fp, n_spaces, 0);
		fprintf (fp, "<ul>\n");
		vftr_make_html_indent (fp, n_spaces, 1);
		fprintf (fp, "<li>\n");
		vftr_print_stacktree_to_html (fp, leaf->callee, n_spaces + 2 * INDENT_SPACES, 0);
		vftr_make_html_indent (fp, n_spaces, 1);
		fprintf (fp, "</li>\n");
		vftr_make_html_indent (fp, n_spaces, 0);
		fprintf (fp, "</ul>\n");
	}

	if (leaf->next_in_level) {
		vftr_make_html_indent (fp, n_spaces, 0);
		fprintf (fp, "<li>\n");
		vftr_print_stacktree_to_html (fp, leaf->next_in_level, n_spaces + INDENT_SPACES, 0);
		vftr_make_html_indent (fp, n_spaces, 0);
		fprintf (fp, "</li>\n");
	}
}

/**********************************************************************/

void vftr_print_html_output (char *func_name, stack_leaf_t *leaf) {
	char html_filename[strlen(func_name) + 6];
	snprintf (html_filename, strlen(func_name) + 6, "%s.html", func_name);
	FILE *fp = fopen (html_filename, "w");
	fprintf (fp, "<h1>Vftrace stack tree for %s</h1>\n", func_name);
	fprintf (fp, "<link rel=\"stylesheet\" href=\"flow.css\">\n");
	fprintf (fp, "<nav class=\"nav\"/>\n");
	vftr_make_html_indent (fp, 0, 1);
	fprintf (fp, "<ul>\n");
	vftr_make_html_indent (fp, 0, 2);
	fprintf (fp, "<li>\n");
	vftr_print_stacktree_to_html (fp, leaf->origin, 3 * INDENT_SPACES, 0);
	vftr_make_html_indent (fp, 0, 2);
	fprintf (fp, "</li>\n");
	vftr_make_html_indent (fp, 0, 1);
	fprintf (fp, "</ul>\n");
	fprintf (fp, "</nav>\n");
	fclose (fp);
}

/**********************************************************************/

void vftr_print_style_file () {
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
	printf ("Normalize stack\n");
	vftr_normalize_stacks();
	printf ("Normalize stack DONE\n");
	fprintf (stdout, "%s: %d %d\n", func2->name, func2->id, func2->gid);
	fprintf (stdout, "%s: %d %d\n", func3->name, func3->id, func3->gid);
	fprintf (stdout, "%s: %d %d\n", func4->name, func4->id, func4->gid);
	fprintf (stdout, "%s: %d %d\n", func5->name, func5->id, func5->gid);
	fprintf (stdout, "%s: %d %d\n", func6->name, func6->id, func6->gid);
	fprintf (stdout, "Global stacklist: \n");
	vftr_print_global_stacklist (stdout);

	int *stack_indices, *func_indices;
	int n_indices;
	vftr_find_function ("C", &func_indices, &stack_indices, &n_indices, false, STACK_INFO);	
 	stack_leaf_t *stack_tree = NULL;
	printf ("n_indices: %d\n", n_indices);
	for (int i = 0;  i < n_indices; i++) {
		//int n_functions_in_stack = vftr_stack_length (stack_indices[i]);
		// Why is n_functions_in_stack 2 instead of 3?
		int n_functions_in_stack = 3;
		printf ("n_function_in_stack: %d\n", n_functions_in_stack);
		int *stack_ids = (int*)malloc (n_functions_in_stack * sizeof(int));	
		int stack_id = stack_indices[i];
		int function_id = func_indices[i];
		printf ("stack_id: %d ", stack_id);
		for (int j = 0; j < n_functions_in_stack; j++) {
			stack_ids[j] = stack_id;
			stack_id = vftr_gStackinfo[stack_id].ret;
			printf ("%d ", stack_id);
		}
		printf ("\n");
		vftr_fill_into_stack_tree (&stack_tree, n_functions_in_stack, stack_ids, function_id);
		free (stack_ids);
	}
	printf ("Stack tree created\n");
	if (stack_tree) {
		printf ("Stack tree exists\n");
	} else {
		printf ("Pointer is zero\n");
	}
	long long dummy1;
	double dummy2;
	vftr_print_stacktree (stdout, stack_tree->origin, 0, &dummy1, &dummy2);
	printf ("Stack tree printed\n");
	vftr_print_html_output ("C", stack_tree->origin);
	free (stack_tree);
	return 0;
}
