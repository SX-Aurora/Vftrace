#include <stdlib.h>
#include <stdio.h>

#include <limits.h>
#include <string.h>
#include <ctype.h>

int vftr_count_base_digits(long long value, int base) {
   int count = 0;
   do {
      count++;
      value /= base;
   } while (value > 0);
   return count;
}

char *vftr_read_file_to_string(char *filename) {
    FILE *file = fopen(filename, "rt");
    if (file == NULL) {
       perror(filename);
       abort();
    }
    fseek(file, 0, SEEK_END);
    size_t length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *buffer = (char *) malloc(length + 1);
    buffer[length] = '\0';
    size_t nread = fread(buffer, 1, length, file);
    if (nread != length) {
       fprintf(stderr, "Unable to read configuration file %s\n"
               "Read %ld bytes, but expected %ld\n",
               filename, nread, length);
       abort();
    }
    fclose(file);
    return buffer;
}

int vftr_levenshtein_kernel(char *a, char *b, int lena, int lenb, int **lookup_table) {
   if (lena == 0) {
      return lenb;
   } else if (lenb == 0) {
      return lena;
   } else if (a[0] == b[0]) {
      if (lookup_table[lena-1][lenb-1] < 0) {
         lookup_table[lena-1][lenb-1] =
            vftr_levenshtein_kernel(a+1, b+1, lena-1, lenb-1, lookup_table);
      }
      return lookup_table[lena-1][lenb-1];
   } else {
      int min = INT_MAX;
      if (lookup_table[lena-1][lenb] < 0) {
         lookup_table[lena-1][lenb] =
            vftr_levenshtein_kernel(a+1, b, lena-1, lenb, lookup_table);
      }
      int lev1 = lookup_table[lena-1][lenb];
      if (lookup_table[lena][lenb-1] < 0) {
         lookup_table[lena][lenb-1] =
            vftr_levenshtein_kernel(a, b+1, lena, lenb-1, lookup_table);
      }
      int lev2 = lookup_table[lena][lenb-1];
      if (lookup_table[lena-1][lenb-1] < 0) {
         lookup_table[lena-1][lenb-1] =
            vftr_levenshtein_kernel(a+1, b+1, lena-1, lenb-1, lookup_table);
      }
      int lev3 = lookup_table[lena-1][lenb-1];
      min = lev1 < lev2 ? lev1 : lev2;
      min = min < lev3 ? min : lev3;
      return 1 + min;
   }
}

int vftr_levenshtein_distance(char *a, char *b) {
   int len1 = strlen(a);
   int len2 = strlen(b);
   // initialize the levenshtein lookup table
   int **lookup_table = (int**) malloc(len1*sizeof(int*));
   for (int i=0; i<len1; i++) {
      lookup_table[i] = (int*) malloc(len2*sizeof(int));
      for (int j=0; j<len2; j++) {
         lookup_table[i][j] = -1;
      }
   }
   int ld = vftr_levenshtein_kernel(a, b, len1, len2, lookup_table);
   // clean up lookup table
   for (int i=0; i<len1; i++) {
      free(lookup_table[i]);
   }
   free(lookup_table);
   return ld;
}

void vftr_chop_trailing_char(char *string, char trailing_char) {
   char *strptr = string;
   // move to the end of the string
   while (*strptr != '\0') {
      strptr++;
   }
   // move backwards eliminating the trailing character
   // if it matches
   while (strptr > string) {
      strptr--;
      if (*strptr == trailing_char) {
         *strptr = '\0';
      } else {
         strptr = string;
      }
   }
}

void vftr_trim_left_with_delimiter(char *string, char *delim) {
   if (string == NULL ||
       delim == NULL ||
       strlen(string) == 0 ||
       strlen(delim)== 0) {
      return;
   }
   char *substring = strstr(string, delim);
   if (substring != NULL) {
      substring += strlen(delim);
      while (*substring != '\0') {
         *string = *substring;
         string++;
         substring++;
      }
      *string = '\0';
   }
}

char *vftr_combine_string_and_address(const char *str, const void *addr) {
   int length = 0;
   length += strlen(str);
   length += snprintf(NULL, 0, "_%p", addr);
   length++; // null terminator
   char *combistring = (char*) malloc(length*sizeof(char));
   snprintf(combistring, length, "%s_%p", str, addr);
   return combistring;
}

char *vftr_byte_unit(unsigned long long size) {
   int i=0;
   while (size > 1024) {
      size /= 1024;
      i++;
   }
   int nunits = 9;
   char *units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"};
   if (i < nunits) {
      return strdup(units[i]);
   }
   return strdup("Unknown Unit");
}
