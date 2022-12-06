#include <stdlib.h>

int main (int argc, char *argv[]) {
    int *a = (int*)malloc(sizeof(int));

    #pragma acc enter data copyin(a) async 
    #pragma acc wait
    int x = 0;
    #pragma acc exit data copyout(a) async
    #pragma acc wait   

    free (a);
}
