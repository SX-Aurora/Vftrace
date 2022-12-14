#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

enum {TEST_SIGTERM = 0,
      TEST_SIGINT = 1,
      TEST_SIGABRT = 2,
      TEST_SIGFPE = 3,
      TEST_SIGQUIT = 4,
      TEST_SIGSEGV = 5};

int main (int argc, char *argv[]) {
   int test_case = argc > 1 ? atoi(argv[1]) : -1;
   switch (test_case) {
      case TEST_SIGTERM:
         raise(SIGTERM);
      case TEST_SIGINT:
         raise(SIGINT);
      case TEST_SIGABRT:
         raise(SIGABRT);
      case TEST_SIGFPE:
         raise(SIGFPE);
      case TEST_SIGQUIT:
         raise(SIGQUIT);
      case TEST_SIGSEGV:
         raise(SIGSEGV);
      default:
         printf ("Unknown test case.\n");
   }
   return 0;
}
