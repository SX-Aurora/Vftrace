#include <stdbool.h>

int vftr_count_base_digits(long long value, int base) {
   int count = 0;
   do {
      count++;
      value /= base;
   } while (value > 0);
   return count;
}

char *vftr_bool_to_string(bool value) {
   return value ? "true" : "false";
}
