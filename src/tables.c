#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <string.h>

#include "tables.h"

// compute column widths
int vftr_column_width_char(int nrows, char *header, char *format, char *values) {
   int width = strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int vftr_column_width_string(int nrows, char *header, char *format, char **values) {
   int width = strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int vftr_column_width_int(int nrows, char *header, char *format, int*values) {
   int width = strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int vftr_column_width_long(int nrows, char *header, char *format, long *values) {
   int width = strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int vftr_column_width_longlong(int nrows, char *header, char *format, long long *values) {
   int width = strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int vftr_column_width_float(int nrows, char *header, char *format, float *values) {
   int width = strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int vftr_column_width_double(int nrows, char *header, char *format, double *values) {
   int width = strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int vftr_column_width_longdouble(int nrows, char *header, char *format, long double *values) {
   int width = strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow]);
      width = width > tmpwidth ? width : tmpwidth;
   }
   return width;
}

int vftr_column_width_bool(int nrows, char *header, char *format, bool *values) {
   int width = strlen(header);
   for (int irow=0; irow<nrows; irow++) {
      int tmpwidth = snprintf(NULL, 0, format, values[irow] ? "true" : "false");
      width = width > tmpwidth ? width : tmpwidth;
   }   
   return width;
}

int *vftr_compute_column_widths(int ncols, int nrows, column_t *coltypes, char **headers,
                                char **formats, void **values) {
   int *colwidths = (int*) malloc(ncols*sizeof(int));
   for (int icol=0; icol<ncols; icol++) {
      switch (coltypes[icol]) {
         case col_char:
            colwidths[icol] = vftr_column_width_char(nrows, headers[icol], formats[icol], (char*) values[icol]);
            break;
         case col_string:
            colwidths[icol] = vftr_column_width_string(nrows, headers[icol], formats[icol], (char**) values[icol]);
            break;
         case col_int:
            colwidths[icol] = vftr_column_width_int(nrows, headers[icol], formats[icol], (int*) values[icol]);
            break;
         case col_long:
            colwidths[icol] = vftr_column_width_long(nrows, headers[icol], formats[icol], (long*) values[icol]);
            break;
         case col_longlong:
            colwidths[icol] = vftr_column_width_longlong(nrows, headers[icol], formats[icol], (long long*) values[icol]);
            break;
         case col_float:
            colwidths[icol] = vftr_column_width_float(nrows, headers[icol], formats[icol], (float*) values[icol]);
            break;
         case col_double:
            colwidths[icol] = vftr_column_width_double(nrows, headers[icol], formats[icol], (double*) values[icol]);
            break;
         case col_longdouble:
            colwidths[icol] = vftr_column_width_longdouble(nrows, headers[icol], formats[icol], (long double*) values[icol]);
            break;
         case col_bool:
            colwidths[icol] = vftr_column_width_bool(nrows, headers[icol], formats[icol], (bool*) values[icol]);
            break;
      }
   }
   return colwidths;
}

// header and decorations
void vftr_print_table_rule(FILE *fp, int ncols, int *widths) {
   fprintf(fp, "+-");
   for (int icol=0; icol<ncols; icol++) {
      for (int i=0; i<widths[icol]; i++) {
         fprintf(fp, "-");
      }
      if (icol == ncols-1) {
         fprintf(fp,"-+\n");
      } else {
         fprintf(fp,"-+-");
      }
   }
}

void vftr_print_table_headers(FILE *fp, int ncols, int *widths, char **headers) {
   fprintf(fp, "| ");
   for (int icol=0; icol<ncols; icol++) {
      int headerlen = strlen(headers[icol]);
      int leftpad = (widths[icol] - headerlen) / 2;
      int rightpad = widths[icol] - headerlen - leftpad;
      fprintf(fp, "%*s", leftpad, "");
      fprintf(fp, "%s", headers[icol]);
      fprintf(fp, "%*s", rightpad, "");
      if (icol == ncols-1) {
         fprintf(fp," |\n");
      } else {
         fprintf(fp," | ");
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
   char *valuestrings[2] = {"true", "false"};
   int idx = value ? 0 : 1;
   int nchars = snprintf(NULL, 0, format, valuestrings[idx]);
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
   fprintf(fp, format, valuestrings[idx]);
   fprintf(fp, "%*s", rightpad, "");
}

void vftr_print_table_values(FILE *fp, int ncols, int nrows, column_t *coltypes, int *colwidths,
                             char **formats, char *align, void **values) {
   for (int irow=0; irow<nrows; irow++) {
      fprintf(fp, "|");
      for (int icol=0; icol<ncols; icol++) {
         fprintf(fp, " ");
         switch (coltypes[icol]) {
            case col_char:
               vftr_print_table_char(fp, colwidths[icol], formats[icol], align[icol], ((char*)values[icol])[irow]);
               break;
            case col_string:
               vftr_print_table_string(fp, colwidths[icol], formats[icol], align[icol], ((char**)values[icol])[irow]);
               break;
            case col_int:
               vftr_print_table_int(fp, colwidths[icol], formats[icol], align[icol], ((int*)values[icol])[irow]);
               break;
            case col_long:
               vftr_print_table_long(fp, colwidths[icol], formats[icol], align[icol], ((long*)values[icol])[irow]);
               break;
            case col_longlong:
               vftr_print_table_longlong(fp, colwidths[icol], formats[icol], align[icol], ((long long*)values[icol])[irow]);
               break;
            case col_float:
               vftr_print_table_float(fp, colwidths[icol], formats[icol], align[icol], ((float*)values[icol])[irow]);
               break;
            case col_double:
               vftr_print_table_double(fp, colwidths[icol], formats[icol], align[icol], ((double*)values[icol])[irow]);
               break;
            case col_longdouble:
               vftr_print_table_longdouble(fp, colwidths[icol], formats[icol], align[icol], ((long double*)values[icol])[irow]);
               break;
            case col_bool:
               vftr_print_table_bool(fp, colwidths[icol], formats[icol], align[icol], ((bool*)values[icol])[irow]);
               break;
         }
         fprintf(fp, " |");
      }
      // linebeak at the end of a column
      fprintf(fp, "\n");
   }
}

void vftr_print_table(FILE *fp, int ncols, int nrows, bool rules[3],
                      column_t *coltypes, char **headers,
                      char **formats, char *align, void **value_lists) {
   // determine the width of each column
   int *colwidths = vftr_compute_column_widths(ncols, nrows, coltypes, headers,
                                               formats, value_lists);

   // print table headers
   if (rules[0]) {vftr_print_table_rule(fp, ncols, colwidths);}
   vftr_print_table_headers(fp, ncols, colwidths, headers);
   if (rules[1]) {vftr_print_table_rule(fp, ncols, colwidths);}

   // print all the rows
   vftr_print_table_values(fp, ncols, nrows, coltypes,colwidths,
                           formats, align, value_lists);
   // print bottomrule
   if (rules[2]) {vftr_print_table_rule(fp, ncols, colwidths);}

   // free memory
   free(colwidths);
}
