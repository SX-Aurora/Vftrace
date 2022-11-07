#include <stdlib.h>
#include <stdbool.h>

#include <ctype.h>
#include <string.h>

char *vftr_bool_to_string(bool value) {
   return value ? "true" : "false";
}

bool vftr_string_to_bool(char *string) {
   bool retval = false;
   if (string != NULL) {
      char *tmpstring = strdup(string);
      char *str = tmpstring;
      while (*str != '\0') {
         *str = tolower(*str);
         str++;
      }
      if (!strcmp(tmpstring, "1") ||
          !strcmp(tmpstring, "yes") ||
          !strcmp(tmpstring, "on")||
          !strcmp(tmpstring, "true")) {
         retval = true;
      } else {
         retval = false;
      }
      free(tmpstring);
   } else {
      retval = false;
   }
   return retval;
}
