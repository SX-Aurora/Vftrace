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

