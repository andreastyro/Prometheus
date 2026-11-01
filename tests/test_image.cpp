#include "ml/data/image.hpp"
#include "ml/utils/testing.hpp"
#include <stdio.h>
#include <string>

using namespace std;

int main(){
    // tests/assets/Andreas.jpg is a 2731 x 2048 RGB JPEG.
    auto img = load_image("tests/assets/Andreas.jpg", true);

    CHECK_NEAR("channels", img->shape[0], 3,    0.0);
    CHECK_NEAR("width",    img->shape[1], 2731, 0.0);
    CHECK_NEAR("height",   img->shape[2], 2048, 0.0);
    CHECK_NEAR("num_el == c*w*h", img->num_el(), 3.0 * 2731 * 2048, 0.0);

    // normalize=true must map bytes into [0, 1]
    float lo = 1e9f, hi = -1e9f;
    for (float v : img->data) { if (v < lo) lo = v; if (v > hi) hi = v; }
    CHECK_TRUE("normalized min >= 0", lo >= 0.0f);
    CHECK_TRUE("normalized max <= 1", hi <= 1.0f);
    CHECK_TRUE("uses a real range, not constant", hi - lo > 0.5f);

    // normalize=false must stay in byte range, and agree with the scaled version
    auto raw = load_image("tests/assets/Andreas.jpg", false);
    CHECK_NEAR("raw matches normalized * 255", raw->data[0], img->data[0] * 255.0, 1.0);
    CHECK_TRUE("raw max > 1", [&]{ float m = 0; for (float v : raw->data) if (v > m) m = v; return m > 1.0f; }());

    bool threw = false;
    try { load_image("tests/assets/definitely_missing.png", true); }
    catch (const std::exception&) { threw = true; }
    CHECK_TRUE("missing file throws", threw);

    return test_report();
}
