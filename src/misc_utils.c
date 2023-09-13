#include <stdlib.h>
#include <stdio.h>

#include <limits.h>
#include <string.h>
#include <ctype.h>

#include "signal_handling.h"

int vftr_count_base_digits(long long value, int base) {
   int count = 0;
   do {
      count++;
      value /= base;
   } while (value > 0);
   return count;
}

int vftr_count_base_digits_float (float value, int base) {
   int count = 0;
   do {
      count++;
      value /= base;
   } while (value > 1);
   return count;
}

char *vftr_read_file_to_string(char *filename) {
    FILE *file = fopen(filename, "rt");
    if (file == NULL) {
       perror(filename);
       vftr_abort(0);
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
       vftr_abort(0);
    }
    fclose(file);
    return buffer;
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
