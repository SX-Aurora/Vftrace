#include <stdio.h>

#include <cupti.h>

int main (int argc, char *argv[]) {
   const char *cbid_name;
   for (int cbid = 0; cbid < 405; cbid++) {
      cuptiGetCallbackName (CUPTI_CB_DOMAIN_RUNTIME_API, cbid, &cbid_name);
      fprintf (stdout, "%d: %s\n", cbid, cbid_name);
   }
   return 0;
}
