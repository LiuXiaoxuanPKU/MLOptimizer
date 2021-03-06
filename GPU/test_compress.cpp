#include <assert.h>     /* assert */
#include <stdlib.h>     /* abs */
#include <iostream>

#include "lz4.hpp"
#include "cascaded.hpp"
#include "nvcomp.hpp"

int main() {
    using T = float;
    int uncompressed_count = 100;
    T uncompressed_data[100];
    T decompressed_data[100];

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < uncompressed_count; i++) {
        uncompressed_data[i] = i;
    }

    T* d_uncompressed_data{nullptr};
    cudaMalloc(&d_uncompressed_data, sizeof(T) * uncompressed_count);
    cudaMemcpy(d_uncompressed_data, uncompressed_data, sizeof(T) * uncompressed_count, cudaMemcpyHostToDevice);


    size_t chunk_size = 1 << 16;
    std::cout << "Start to Compress" << std::endl;
    LZ4Compressor<T> compressor(d_uncompressed_data, uncompressed_count, chunk_size);

    const size_t compress_temp_size = compressor.get_temp_size();
    void * compress_temp_space;
    cudaMalloc(&compress_temp_space, compress_temp_size);

    size_t compress_output_size = compressor.get_max_output_size(
        compress_temp_space, compress_temp_size);
    void * compress_output_space;
    cudaMalloc(&compress_output_space, compress_output_size);

    nvcompError_t status;
    compressor.compress_async(compress_temp_space,
        compress_temp_size, compress_output_space, &compress_output_size, stream);

    cudaStreamSynchronize(stream);
    cudaFree(compress_temp_space);
    cudaFree(d_uncompressed_data);


    std::cout << "Start to Decompress" << std::endl;
    nvcomp::Decompressor<T> decompressor(compress_output_space, compress_output_size, stream);
    const size_t decompress_temp_size = decompressor.get_temp_size();
    void * decompress_temp_space;
    cudaMalloc(&decompress_temp_space, decompress_temp_size);
    const size_t decompress_output_count = decompressor.get_num_elements();
    T * decompress_output_space;
    cudaMalloc((void**)&decompress_output_space, decompress_output_count*sizeof(T));
    decompressor.decompress_async(decompress_temp_space, decompress_temp_size,
                                    decompress_output_space, decompress_output_count,
                                    stream);

    cudaStreamSynchronize(stream);


    cudaMemcpy(decompressed_data, decompress_output_space, decompress_output_count, cudaMemcpyDeviceToHost);

    cudaFree(compress_output_space);
    cudaFree(decompress_temp_space);
    cudaFree(decompress_output_space);

    for (int i = 0; i < uncompressed_count; i++) {
        assert(abs(uncompressed_data[i] - decompressed_data[i]) < 1e-6);
    }

}