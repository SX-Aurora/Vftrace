__attribute__((no_instrument_function))
int not_instrumented_function () {
   return 0;
}
int main() {
   not_instrumented_function();
}
