#include <stdlib.h>

int main (int argc, char *argv[]) {
    int *a = (int*)malloc(sizeof(int));
    int *b = (int*)malloc(sizeof(int));

    #pragma acc enter data copyin(a)
    #pragma acc enter data copyin(b)
    int x;
    #pragma acc exit data copyout(a)
    #pragma acc exit data copyout(b)

    free (a);
    free (b);
}
