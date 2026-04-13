#define STB_IMAGE_IMPLEMENTATION

#include "ml/data/stb_image.h"
#include "ml/data/image.hpp"
#include "ml/tensor.hpp"

using namespace std;

TensorPtr load_image(const string& path, bool normalize){

    int width, height, channels;

    unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 0);

    vector<int> tensor_shape = {channels, width, height};

    auto tensor = make_shared<Tensor>(tensor_shape);

    for (int c = 0; c < channels; c++){
        for (int w = 0; w < width; w++){
            for (int h = 0; h < height; h++){
                int source = (h * width + w) * channels + c;
                int destination = c * width * height + w * height + h;

                float val = img[source];

                if (normalize) val /= 255.0f;
                tensor->data[destination] = val;

            }
        }
    }

    stbi_image_free(img);
    return tensor;

}