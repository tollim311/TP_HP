#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Fonction Gaussienne (calculée sur CPU)
__global__ double gaussian(double x, double sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

// Filtre bilatéral
__global__ void bilateral_filter_cuda(unsigned char *src, unsigned char *dst, int width, int height, int channels, 
                                      int d, double sigma_color, double sigma_space, double *spatial_weights) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = d / 2;

    if (x >= radius && x < width - radius && y >= radius && y < height - radius) {
        
        unsigned char *center_pixel = src + (y * width + x) * channels;

        double weight_sum[3] = {0.0, 0.0, 0.0};
        double filtered_value[3] = {0.0, 0.0, 0.0};

        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                int nx = x + j - radius;
                int ny = y + i - radius;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int neighbor_index = (ny * width + nx) * channels;
                    unsigned char *neighbor_pixel = &src[neighbor_index];

                    for (int c = 0; c < channels; c++) {
                        double range_weight = gaussian(abs(neighbor_pixel[c] - center_pixel[c]), sigma_color);
                        double weight = spatial_weights[i * d + j] * range_weight;

                        filtered_value[c] += neighbor_pixel[c] * weight;
                        weight_sum[c] += weight;
                    }
                }
            }
        }

        for (int c = 0; c < channels; c++) {
            dst[(y * width + x) * channels + c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6)); // Évite la division par 0
        }
    }
}

// Fonction principale
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!image) {
        printf("Erreur de chargement de l'image !\n");
        return 1;
    }

    int d = 5;
    double sigma_color = 75.0, sigma_space = 75.0;
    int radius = d / 2;

    // Calcul des poids spatiaux (en CPU, car ils sont fixes évite un grand coup de calcul sur GPU)
    double *spatial_weights = (double *)malloc(d * d * sizeof(double));
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            int x = i - radius, y = j - radius;
            spatial_weights[i * d + j] = gaussian(sqrt(x * x + y * y), sigma_space);
        }
    }

    // Allocation mémoire pour l'image de sortie
    unsigned char *filtered_image = (unsigned char *)malloc(width * height * channels);
    if (!filtered_image) {
        printf("Erreur d'allocation mémoire pour l'image filtrée !\n");
        free(spatial_weights);
        return 1;
    }

    // Allocation mémoire sur GPU
    unsigned char *d_src, *d_dst;
    double *d_spatial_weights;
    cudaMalloc(&d_src, width * height * channels);
    cudaMalloc(&d_dst, width * height * channels);
    cudaMalloc(&d_spatial_weights, d * d * sizeof(double));

    // Copier les données vers le GPU
    cudaMemcpy(d_src, image, width * height * channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_spatial_weights, spatial_weights, d * d * sizeof(double), cudaMemcpyHostToDevice);

    // Définir la grille et les blocs
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Lancer le noyau CUDA
    bilateral_filter_cuda<<<gridSize, blockSize>>>(d_src, d_dst, width, height, channels, d, sigma_color, sigma_space, d_spatial_weights);
    cudaDeviceSynchronize();

    // Copier les résultats vers le CPU
    cudaMemcpy(filtered_image, d_dst, width * height * channels, cudaMemcpyDeviceToHost);

    // Sauvegarder l'image filtrée
    if (!stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels)) {
        printf("Erreur lors de l'enregistrement de l'image !\n");
    } else {
        printf("Filtrage bilatéral terminé. Image enregistrée sous %s\n", argv[2]);
    }

    // Libérer la mémoire du GPU et du CPU
    stbi_image_free(image);
    free(filtered_image);
    free(spatial_weights);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_spatial_weights);

    return 0;
}
