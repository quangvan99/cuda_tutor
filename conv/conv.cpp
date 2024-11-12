#include <vector>
#include <cmath>
#include <iostream>
// Helper function to perform 2D convolution
std::vector<std::vector<float>> convolution2D(
    const std::vector<std::vector<float>>& image,
    const std::vector<std::vector<float>>& kernel) {
    
    int image_height = image.size();
    int image_width = image[0].size();
    int kernel_height = kernel.size();
    int kernel_width = kernel[0].size();
    
    // Calculate padding
    int pad_h = kernel_height / 2;
    int pad_w = kernel_width / 2;
    
    // Output dimensions
    std::vector<std::vector<float>> output(
        image_height,
        std::vector<float>(image_width, 0.0f)
    );
    
    // Perform convolution
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            float sum = 0.0f;
            
            // Apply kernel
            for (int k = -pad_h; k <= pad_h; k++) {
                for (int l = -pad_w; l <= pad_w; l++) {
                    int img_row = i + k;
                    int img_col = j + l;
                    
                    // Check boundaries
                    if (img_row >= 0 && img_row < image_height && 
                        img_col >= 0 && img_col < image_width) {
                        sum += image[img_row][img_col] * 
                               kernel[k + pad_h][l + pad_w];
                    }
                }
            }
            output[i][j] = sum;
        }
    }
    
    return output;
}

int main() {
    // Example usage
    // Create a sample 5x5 image
    std::vector<std::vector<float>> image = {
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25}
    };
    
    // Create a 3x3 kernel (e.g., Gaussian blur)
    std::vector<std::vector<float>> kernel = {
        {1.0f/16, 2.0f/16, 1.0f/16},
        {2.0f/16, 4.0f/16, 2.0f/16},
        {1.0f/16, 2.0f/16, 1.0f/16}
    };
    
    // Perform convolution
    auto result = convolution2D(image, kernel);
    
    // Print result
    for (const auto& row : result) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}
