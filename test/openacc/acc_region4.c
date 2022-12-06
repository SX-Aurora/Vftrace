#include <stdlib.h>

int main (int argc, char *argv[]) {
    int *a = (int*)malloc(sizeof(int));
    int *b = (int*)malloc(sizeof(int));

    #pragma acc enter data copyin(a) async(1)
    #pragma acc enter data copyin(b) async(2)
    #pragma acc wait(2)
    int x2 = 0; 
    #pragma acc wait(1)
    #pragma acc exit data copyout(a)
    #pragma acc exit data copyout(b)

    free (a);
    free (b);
}
