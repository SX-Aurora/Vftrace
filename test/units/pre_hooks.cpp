#include <iostream>
using namespace std;

int before_main(void) {
  cout<<"before main\n";
  return 137;
}

static int dummy = before_main(); 

int main() {
  cout<<"inside main\n";
  cout<<dummy<<"\n";
  return 0;
}
