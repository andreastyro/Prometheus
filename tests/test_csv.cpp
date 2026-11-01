#include "ml/data/csv.hpp"
#include "ml/utils/testing.hpp"
#include <stdio.h>
#include <string>

using namespace std;

int main(){
    // tests/test.csv: header + 4 rows of feature1,feature2,feature3,label
    //   1, 2, 3, 0
    //   4, 5, 6, 1
    //   7, 8, 9, 0
    //  10,11,12, 1
    //  13,14,15, 0
    auto [x, y] = read_csv("tests/test.csv", -1, true);

    CHECK_NEAR("x rows", x->shape[0], 5, 0.0);
    CHECK_NEAR("x cols", x->shape[1], 3, 0.0);
    CHECK_NEAR("y rows", y->shape[0], 5, 0.0);
    CHECK_NEAR("y cols", y->shape[1], 1, 0.0);

    CHECK_NEAR("x[0,0]", x->data[0],  1.0, 1e-6);
    CHECK_NEAR("x[0,2]", x->data[2],  3.0, 1e-6);
    CHECK_NEAR("x[3,0]", x->data[9],  10.0, 1e-6);
    CHECK_NEAR("x[3,2]", x->data[11], 12.0, 1e-6);

    CHECK_NEAR("y[0]", y->data[0], 0.0, 1e-6);
    CHECK_NEAR("y[1]", y->data[1], 1.0, 1e-6);
    CHECK_NEAR("y[3]", y->data[3], 1.0, 1e-6);
    CHECK_NEAR("x[4,0]", x->data[12], 13.0, 1e-6);
    CHECK_NEAR("x[4,2]", x->data[14], 15.0, 1e-6);
    CHECK_NEAR("y[4]", y->data[4], 0.0, 1e-6);

    // a missing file must raise, not crash or return empty tensors
    bool threw = false;
    try { read_csv("tests/definitely_missing.csv", -1, true); }
    catch (const std::exception&) { threw = true; }
    CHECK_TRUE("missing file throws", threw);

    return test_report();
}
