#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <string.h>

#include "table_types.h"
#include "tables.h"
#include "misc_utils.h"

// compute column widths
int vftr_column_width_char(int nrows, char *header, char *format, char *values) {
   int width = header == NULL ? 0 : strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int vftr_column_width_string(int nrows, char *header, char *format, char **values) {
   int width = header == NULL ? 0 : strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int vftr_column_width_int(int nrows, char *header, char *format, int*values) {
   int width = header == NULL ? 0 : strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int vftr_column_width_long(int nrows, char *header, char *format, long *values) {
   int width = header == NULL ? 0 : strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int vftr_column_width_longlong(int nrows, char *header, char *format, long long *values) {
   int width = header == NULL ? 0 : strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int vftr_column_width_float(int nrows, char *header, char *format, float *values) {
   int width = header == NULL ? 0 : strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int vftr_column_width_double(int nrows, char *header, char *format, double *values) {
   int width = header == NULL ? 0 : strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int vftr_column_width_longdouble(int nrows, char *header, char *format, long double *values) {
   int width = header == NULL ? 0 : strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }
   return width;
}

int vftr_column_width_bool(int nrows, char *header, char *format, bool *values) {
   int width = header == NULL ? 0 : strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, vftr_bool_to_string(values[irow]));
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int *vftr_compute_column_widths(int ncols, int nrows, column_t *columns) {
   int *colwidths = (int*) malloc(ncols*sizeof(int));
   for (int icol=0; icol<ncols; icol++) {
      switch (columns[icol].kind) {
         case col_char:
            colwidths[icol] = vftr_column_width_char(nrows, columns[icol].header, columns[icol].format, (char*) columns[icol].values);
            break;
         case col_string:
            colwidths[icol] = vftr_column_width_string(nrows, columns[icol].header, columns[icol].format, (char**) columns[icol].values);
            break;
         case col_int:
            colwidths[icol] = vftr_column_width_int(nrows, columns[icol].header, columns[icol].format, (int*) columns[icol].values);
            break;
         case col_long:
            colwidths[icol] = vftr_column_width_long(nrows, columns[icol].header, columns[icol].format, (long*) columns[icol].values);
            break;
         case col_longlong:
            colwidths[icol] = vftr_column_width_longlong(nrows, columns[icol].header, columns[icol].format, (long long*) columns[icol].values);
            break;
         case col_float:
            colwidths[icol] = vftr_column_width_float(nrows, columns[icol].header, columns[icol].format, (float*) columns[icol].values);
            break;
         case col_double:
            colwidths[icol] = vftr_column_width_double(nrows, columns[icol].header, columns[icol].format, (double*) columns[icol].values);
            break;
         case col_longdouble:
            colwidths[icol] = vftr_column_width_longdouble(nrows, columns[icol].header, columns[icol].format, (long double*) columns[icol].values);
            break;
         case col_bool:
            colwidths[icol] = vftr_column_width_bool(nrows, columns[icol].header, columns[icol].format, (bool*) columns[icol].values);
            break;
         case col_none:
         default:
            fprintf(stderr, "Unknown column type for table");
      }
   }
   return colwidths;
}

// header and decorations
void vftr_print_table_hline(FILE *fp, int ncols, bool vlines[3], int *widths) {
   char crosschars[3];
   for (int ivline=0; ivline<3; ivline++) {
      crosschars[ivline] = vlines[ivline] ? '+' : '-';
   }
   fprintf(fp, "%c-", crosschars[0]);
   for (int icol=0; icol<ncols; icol++) {
      for (int i=0; i<widths[icol]; i++) {
         fprintf(fp, "-");
      }
      if (icol == ncols-1) {
         fprintf(fp,"-%c\n", crosschars[2]);
      } else {
         fprintf(fp,"-%c-", crosschars[1]);
      }
   }
}

bool vftr_table_headers_are_empty(int ncols, column_t *columns) {
   bool empty = true;
   for (int icol=0; icol<ncols; icol++) {
      empty = empty && columns[icol].header == NULL;
   }
   return empty;
}

void vftr_print_table_headers(FILE *fp, int ncols, int *widths,
                              bool vlines[3], column_t *columns) {
   char vlinechar[3];
   for (int ivline=0; ivline<3; ivline++) {
      vlinechar[ivline] = vlines[ivline] ? '|' : ' ';
   }
   fprintf(fp, "%c ", vlinechar[0]);
   for (int icol=0; icol<ncols; icol++) {
      int headerlen = columns[icol].header == NULL ? 0 : strlen(columns[icol].header);
      int leftpad;
      switch (columns[icol].headeralign) {
         case 'l':
            leftpad = 0;
            break;
         case 'r':
            leftpad = widths[icol] - headerlen;
            break;
         case 'c':
         default:
            leftpad = (widths[icol] - headerlen) / 2;
      }
      int rightpad = widths[icol] - headerlen - leftpad;
      fprintf(fp, "%*s", leftpad, "");
      fprintf(fp, "%s", columns[icol].header == NULL ? "" : columns[icol].header);
      fprintf(fp, "%*s", rightpad, "");
      if (icol == ncols-1) {
         fprintf(fp," %c\n", vlinechar[2]);
      } else {
         fprintf(fp," %c ", vlinechar[1]);
      }
   }
}

void vftr_print_table_char(FILE *fp, int width, char *format, char align, char value) {
   int nchars = snprintf(NULL, 0, format, value);
   int leftpad;
   switch (align) {
      case 'l':
         leftpad = 0;
         break;
      case 'c':
         leftpad = (width - nchars) / 2;
         break;
      case 'r':
      default:
         leftpad = width - nchars;
   }
   int rightpad = width - nchars - leftpad;
   fprintf(fp, "%*s", leftpad, "");
   fprintf(fp, format, value);
   fprintf(fp, "%*s", rightpad, "");
}

void vftr_print_table_string(FILE *fp, int width, char *format, char align, char *value) {
   int nchars = snprintf(NULL, 0, format, value);
   int leftpad;
   switch (align) {
      case 'l':
         leftpad = 0;
         break;
      case 'c':
         leftpad = (width - nchars) / 2;
         break;
      case 'r':
      default:
         leftpad = width - nchars;
   }
   int rightpad = width - nchars - leftpad;
   fprintf(fp, "%*s", leftpad, "");
   fprintf(fp, format, value);
   fprintf(fp, "%*s", rightpad, "");
}

void vftr_print_table_int(FILE *fp, int width, char *format, char align, int value) {
   int nchars = snprintf(NULL, 0, format, value);
   int leftpad;
   switch (align) {
      case 'l':
         leftpad = 0;
         break;
      case 'c':
         leftpad = (width - nchars) / 2;
         break;
      case 'r':
      default:
         leftpad = width - nchars;
   }
   int rightpad = width - nchars - leftpad;
   fprintf(fp, "%*s", leftpad, "");
   fprintf(fp, format, value);
   fprintf(fp, "%*s", rightpad, "");
}

void vftr_print_table_long(FILE *fp, int width, char *format, char align, long value) {
   int nchars = snprintf(NULL, 0, format, value);
   int leftpad;
   switch (align) {
      case 'l':
         leftpad = 0;
         break;
      case 'c':
         leftpad = (width - nchars) / 2;
         break;
      case 'r':
      default:
         leftpad = width - nchars;
   }
   int rightpad = width - nchars - leftpad;
   fprintf(fp, "%*s", leftpad, "");
   fprintf(fp, format, value);
   fprintf(fp, "%*s", rightpad, "");
}

void vftr_print_table_longlong(FILE *fp, int width, char *format, char align, long long value) {
   int nchars = snprintf(NULL, 0, format, value);
   int leftpad;
   switch (align) {
      case 'l':
         leftpad = 0;
         break;
      case 'c':
         leftpad = (width - nchars) / 2;
         break;
      case 'r':
      default:
         leftpad = width - nchars;
   }
   int rightpad = width - nchars - leftpad;
   fprintf(fp, "%*s", leftpad, "");
   fprintf(fp, format, value);
   fprintf(fp, "%*s", rightpad, "");
}

void vftr_print_table_float(FILE *fp, int width, char *format, char align, float value) {
   int nchars = snprintf(NULL, 0, format, value);
   int leftpad;
   switch (align) {
      case 'l':
         leftpad = 0;
         break;
      case 'c':
         leftpad = (width - nchars) / 2;
         break;
      case 'r':
      default:
         leftpad = width - nchars;
   }
   int rightpad = width - nchars - leftpad;
   fprintf(fp, "%*s", leftpad, "");
   fprintf(fp, format, value);
   fprintf(fp, "%*s", rightpad, "");
}

void vftr_print_table_double(FILE *fp, int width, char *format, char align, double value) {
   int nchars = snprintf(NULL, 0, format, value);
   int leftpad;
   switch (align) {
      case 'l':
         leftpad = 0;
         break;
      case 'c':
         leftpad = (width - nchars) / 2;
         break;
      case 'r':
      default:
         leftpad = width - nchars;
   }
   int rightpad = width - nchars - leftpad;
   fprintf(fp, "%*s", leftpad, "");
   fprintf(fp, format, value);
   fprintf(fp, "%*s", rightpad, "");
}

void vftr_print_table_longdouble(FILE *fp, int width, char *format, char align, long double value) {
   int nchars = snprintf(NULL, 0, format, value);
   int leftpad;
   switch (align) {
      case 'l':
         leftpad = 0;
         break;
      case 'c':
         leftpad = (width - nchars) / 2;
         break;
      case 'r':
      default:
         leftpad = width - nchars;
   }
   int rightpad = width - nchars - leftpad;
   fprintf(fp, "%*s", leftpad, "");
   fprintf(fp, format, value);
   fprintf(fp, "%*s", rightpad, "");
}

void vftr_print_table_bool(FILE *fp, int width, char *format, char align, bool value) {
   char *valuestring = vftr_bool_to_string(value);
   int nchars = snprintf(NULL, 0, format, valuestring);
   int leftpad;
   switch (align) {
      case 'l':
         leftpad = 0;
         break;
      case 'c':
         leftpad = (width - nchars) / 2;
         break;
      case 'r':
      default:
         leftpad = width - nchars;
   }
   int rightpad = width - nchars - leftpad;
   fprintf(fp, "%*s", leftpad, "");
   fprintf(fp, format, valuestring);
   fprintf(fp, "%*s", rightpad, "");
}

void vftr_print_table_values(FILE *fp, table_t table, int *colwidths) {

   char vlinechars[3];
   for (int ivline=0; ivline<3; ivline++) {
      vlinechars[ivline] = table.vlines[ivline] ? '|' : ' ';
   }

   int nrows = table.nrows;
   int ncols = table.ncols;
   for (int irow=0; irow<nrows; irow++) {
      fprintf(fp, "%c", vlinechars[0]);
      for (int icol=0; icol<ncols; icol++) {
         column_t column = table.columns[icol];
         fprintf(fp, " ");
         switch (column.kind) {
            case col_char:
               vftr_print_table_char(fp, colwidths[icol], column.format, column.align, ((char*) column.values)[irow]);
               break;
            case col_string:
               vftr_print_table_string(fp, colwidths[icol], column.format, column.align, ((char**) column.values)[irow]);
               break;
            case col_int:
               vftr_print_table_int(fp, colwidths[icol], column.format, column.align, ((int*) column.values)[irow]);
               break;
            case col_long:
               vftr_print_table_long(fp, colwidths[icol], column.format, column.align, ((long*) column.values)[irow]);
               break;
            case col_longlong:
               vftr_print_table_longlong(fp, colwidths[icol], column.format, column.align, ((long long*) column.values)[irow]);
               break;
            case col_float:
               vftr_print_table_float(fp, colwidths[icol], column.format, column.align, ((float*) column.values)[irow]);
               break;
            case col_double:
               vftr_print_table_double(fp, colwidths[icol], column.format, column.align, ((double*) column.values)[irow]);
               break;
            case col_longdouble:
               vftr_print_table_longdouble(fp, colwidths[icol], column.format, column.align, ((long double*) column.values)[irow]);
               break;
            case col_bool:
               vftr_print_table_bool(fp, colwidths[icol], column.format, column.align, ((bool*) column.values)[irow]);
               break;
            case col_none:
            default:
               fprintf(stderr, "Unknown column type for table");
         }

         if (icol == ncols-1) {
            fprintf(fp," %c\n", vlinechars[2]);
         } else {
            fprintf(fp," %c", vlinechars[1]);
         }
      }
      if (table.hlines[2] && irow < nrows-1) {
         vftr_print_table_hline(fp, table.ncols, table.vlines, colwidths);
      }
   }
}

void vftr_print_table(FILE *fp, table_t table) {
   // determine the width of each column
   int *colwidths = vftr_compute_column_widths(table.ncols, table.nrows, table.columns);

   // print table headers
   if (table.hlines[0]) {
      vftr_print_table_hline(fp, table.ncols, table.vlines, colwidths);
   }
   if (!vftr_table_headers_are_empty(table.ncols, table.columns)) {
      // header is considered empty if it only contains null pointers;
      vftr_print_table_headers(fp, table.ncols, colwidths,
                               table.vlines, table.columns);
      if (table.hlines[1]) {
         vftr_print_table_hline(fp, table.ncols, table.vlines, colwidths);
      }
   }

   // print all the rows
   vftr_print_table_values(fp, table, colwidths);

   // print bottom hline
   if (table.hlines[3]) {
      vftr_print_table_hline(fp, table.ncols, table.vlines, colwidths);
   }

   // free memory
   free(colwidths);
}

column_t vftr_new_column() {
   column_t column = {
      .kind = col_none,
      .header = NULL,
      .format = NULL,
      .headeralign = 'c',
      .align = 'r',
      .values = NULL
   };
   return column;
}

table_t vftr_new_table() {
   table_t table = {
      .ncols = 0,
      .nrows = 0,
      .hlines = {true, true, false, true},
      .vlines = {true, true, true},
      .columns = NULL
   };
   return table;
}

void vftr_table_free(table_t *table_ptr) {
   if (table_ptr->ncols > 0) {
      free(table_ptr->columns);
   }
   *table_ptr = vftr_new_table();
}

void vftr_table_set_nrows(table_t *table_ptr, int nrows) {
   table_ptr->nrows = nrows;
}

void vftr_table_add_column(table_t *table_ptr, column_kind kind,
                           char *header, char *format,
                           char headeralign, char align,
                           void *values) {
   int idx = table_ptr->ncols;
   int ncols = table_ptr->ncols+1;
   table_ptr->columns = (column_t*) realloc(table_ptr->columns, ncols*sizeof(column_t));

   table_ptr->columns[idx].kind = kind;
   table_ptr->columns[idx].header = header;
   table_ptr->columns[idx].format = format;
   table_ptr->columns[idx].headeralign = headeralign;
   table_ptr->columns[idx].align = align;
   table_ptr->columns[idx].values = values;
   table_ptr->ncols = ncols;
}

void vftr_table_left_outline(table_t *table, bool value) {
   table->vlines[0] = value;
}

void vftr_table_right_outline(table_t *table, bool value) {
   table->vlines[2] = value;
}

void vftr_table_columns_separating_line(table_t *table, bool value) {
   table->vlines[1] = value;
}

void vftr_table_top_outline(table_t *table, bool value) {
   table->hlines[0] = value;
}

void vftr_table_bottom_outline(table_t *table, bool value) {
   table->hlines[3] = value;
}

void vftr_table_header_separating_line(table_t *table, bool value) {
   table->hlines[1] = value;
}

void vftr_table_rows_separating_line(table_t *table, bool value) {
   table->hlines[2] = value;
}

