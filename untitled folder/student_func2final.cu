#include "reference_calc.cpp"
#include "utils.h"

    __global__
    void gaussian_blur(const unsigned char* const inputChannel,
                       unsigned char* const outputChannel,
                       int numRows, int numCols,
                       const float* const filter, const int filterWidth)
    {
        int absolute_image_position_x = blockIdx.x * blockDim.x + threadIdx.x;
        int absolute_image_position_y = blockIdx.y * blockDim.y + threadIdx.y;
        if (absolute_image_position_x >= numCols || absolute_image_position_y >= numRows)
        {
            return;
        }
        int imageIndex = + absolute_image_position_x + absolute_image_position_y * numCols;
        float sum = 0.0;
        for (int i = 0; i < filterWidth; ++i)
        {
            for (int j = 0; j < filterWidth; ++j)
            {
                int filtered_x = absolute_image_position_x + i - (filterWidth/2);
                int filtered_y = absolute_image_position_y + j - (filterWidth/2);
                if (filtered_x < 0) {
                    filtered_x = 0; // min before going to low
                }
                if (filtered_x >= numCols){
                    filtered_x = numCols - 1; // max before going to high
                }
                if (filtered_y < 0) {
                    filtered_y = 0; // min before going to low
                }
                if (filtered_y >= numRows){
                    filtered_y = numRows - 1; // max before going to high
                }
                sum += float(inputChannel[filtered_y * numCols + filtered_x]) * filter[j * filterWidth + i];
            }
        }
        outputChannel[imageIndex] = char(sum);
}

__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  // TODO
  //
  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  int absolute_image_position_x = blockIdx.x * blockDim.x + threadIdx.x;
  int absolute_image_position_y = blockIdx.y * blockDim.y + threadIdx.y;
  if ( absolute_image_position_x >= numCols || absolute_image_position_y >= numRows ) {
      return;
  }
  // after check, put them all into one color channel for each
  int imageIndex = numCols * absolute_image_position_y + absolute_image_position_x;
  redChannel[imageIndex] = inputImageRGBA[imageIndex].x; // X -> Red
  greenChannel[imageIndex] = inputImageRGBA[imageIndex].y; // Y -> Gree
  blueChannel[imageIndex] = inputImageRGBA[imageIndex].z; // Z -> Blu
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  // R
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  // G
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  // B
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));
    //  filter
  checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));
  // filter copy
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));


}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  //TODO: Set reasonable block size (i.e., number of threads per block)
  int N = 32; // N in N*N -> change here to change value of N
  // works for 16 too
  const dim3 blockSize(N,N,1); // 32 * 32 block size - 2 dimensional


  //TODO:
  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  int tempRow = numRows/ blockSize.y +1;
  int tempCol = numCols/ blockSize.x +1;
  const dim3 gridSize(numCols/ blockSize.x +1,numRows/ blockSize.y +1,1); // - 2 dimensional

  //TODO: Launch a kernel for separating the RGBA image into different color channels
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); // because you told me to

  //TODO: Call your convolution kernel here 3 times, once for each color channel.

  // RED
  gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); // because you told me to
  // GREEN
  gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); // because you told me to

  // BLUE
  gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter,filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); // because you told me to

  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
