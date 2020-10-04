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

void vftr_print_stacktree_to_html (FILE *fp, stack_leaf_t *leaf, int n_spaces, long long *total_time) {
	if (!leaf) return;
	for (int i = 0; i < n_spaces; i++) fprintf (fp, " ");
	fprintf (fp, "<a hfref=\"#\">%s</a>\n", vftr_gStackinfo[leaf->stack_id].name);
	if (leaf->callee) {
		for (int i = 0; i < n_spaces; i++) fprintf (fp, " ");
		fprintf (fp, "<ul>\n");
		for (int i = 0; i < n_spaces + 3; i++) fprintf (fp, " ");
		fprintf (fp, "<li>\n");
		vftr_print_stacktree_to_html (fp, leaf->callee, n_spaces + 6, 0);
		for (int i = 0; i < n_spaces + 3; i++) fprintf (fp, " ");
		fprintf (fp, "</li>\n");
		for (int i = 0; i < n_spaces; i++) fprintf (fp, " ");
		fprintf (fp, "</ul>\n");
	} else {
	}
	if (leaf->next_in_level) {
		for (int i = 0; i < n_spaces; i++) fprintf (fp, " ");
		fprintf (fp, "<li>\n");
		vftr_print_stacktree_to_html (fp, leaf->next_in_level, n_spaces + 3, 0);
		for (int i = 0; i < n_spaces; i++) fprintf (fp, " ");
		fprintf (fp, "</li>\n");
	}
}

/**********************************************************************/

void vftr_print_html_output (char *func_name, stack_leaf_t *leaf) {
	char html_filename[strlen(func_name) + 6];
	snprintf (html_filename, strlen(func_name) + 6, "%s.html", func_name);
	FILE *fp = fopen (html_filename, "w");
	fprintf (fp, "HUHU\n");
	fclose (fp);
}

