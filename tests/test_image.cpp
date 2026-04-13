#include "ml/data/image.hpp"
#include <stdio.h>

using namespace std;

int main(){

    auto img = load_image("tests/assets/Andreas.jpg", true);

    printf("Shape: [%d, %d, %d]\n", img->shape[0], img->shape[1], img->shape[2]);
    printf("Channels: %d\n", img->shape[0]);
    printf("Width:    %d\n", img->shape[1]);
    printf("Height:   %d\n", img->shape[2]);
    printf("Total pixels: %d\n", img->num_el());

    // print first 9 pixel values (first 3 pixels, all channels)
    printf("\nFirst 3 pixels (R, G, B):\n");
    for (int p = 0; p < 3; p++){
        float r = img->data[0 * img->shape[1] * img->shape[2] + p];
        float g = img->data[1 * img->shape[1] * img->shape[2] + p];
        float b = img->data[2 * img->shape[1] * img->shape[2] + p];
        printf("Pixel %d: R=%.3f G=%.3f B=%.3f\n", p, r, g, b);
    }

    return 0;
}
