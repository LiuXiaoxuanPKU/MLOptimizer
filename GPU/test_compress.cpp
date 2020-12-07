#include <math>
#include <iostream>

#include "cascaded.hpp"
#include "nvcomp.hpp"

int uncompressed_count = 100;
float uncompressed_data[100];
float h_compressed_data[100];

for (int i = 0; i < uncompressed_count; i++) {
    uncompressed_data[i] = i;
}

float* d_uncompressed_data{nullptr};
cudaMalloc(&d_uncompressed_data, sizeof(float) * uncompressed_count);
cudaMemcpy(d_uncompressed_data, uncompressed_data, sizeof(uncompressed_data), cudaMemcpyHostToDevice);


std::cout << "Start to Compress" << std::endl;
nvcomp::CascadedCompressor<float> compressor(
    uncompressed_data, uncompressed_count, 2, 1, true);

const size_t compress_temp_size = compressor.get_temp_size();
void * compress_temp_space;
cudaMalloc(&compress_temp_space, compress_temp_size);

size_t output_size = compressor.get_max_output_size(
    temp_space, temp_size);
void * output_space;
cudaMalloc(&output_space, output_size);

nvcompError_t status;
compressor.compress_async(temp_space,
    temp_size, output_space, &output_size, stream);


std::cout << "Start to Decompress" << std::endl;
nvcomp::Decompressor<float> decompressor(compressed_data, output_size, stream);
const size_t decompress_temp_size = decompressor.get_temp_size();
void * decompress_temp_space;
cudaMalloc(&decompress_temp_space, decompress_temp_size);
const size_t decompress_output_count = decompressor.get_num_elements();
int * decompress_output_space;
cudaMalloc((void**)&decompress_output_space, decompress_output_count*sizeof(float));
decompressor.decompress_async(decompress_temp_space, decompress_temp_size,
                                decompress_output_space, decompress_output_count,
                                stream);


cudaMemcpy(decompressed_data, decompress_output_space, decompress_output_count, cudaMemcpyDeviceToHost);
for (int i = 0; i < uncompressed_count; i++) {
    assert(abs(uncompressed_data[i] - decompressed_data[i]) < 1e-6);
}