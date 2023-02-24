#include <stdlib.h>

int main (int argc, char *argv[]) {
    int N = 10;
    int a[N];

    #pragma acc enter data copyin(a)
    #pragma acc exit data copyout(a)
}
