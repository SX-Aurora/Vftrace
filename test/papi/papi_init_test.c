#include <stdio.h>
#include <papi.h>

int main (int argc, char *argv[]) {
   if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
       printf ("FAIL: Init PAPI library\n");
       return -1;
   } else {
       printf ("SUCCESS: Init PAPI library\n"); 
       return 0;
   }
}
