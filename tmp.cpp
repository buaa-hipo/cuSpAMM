#include <iostream>
using namespace std;
template <typename T> void func(T t) { cout << t << endl; }
template <typename T> void compare() {
    // const T& left, const T& right
    // if (left < right) {
    //     return -1;
    // }
    // if (right < left) {
    //     return 1;
    // }
    // return 0;
}

void invoke(void (*p)(int)) {
    int num = 10;
    p(num);
}
int main() { invoke(func); }