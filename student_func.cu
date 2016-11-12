//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */

#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"
#include <cstdio>


/*** DEBUG FUNCTIONS ***/
void printDeviceArray(bool* d_array, int numRows, int numCols)
{

  int size = numRows * numCols;
  bool* h_array = new bool[size];
  cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

  for(int i = 0; i < numRows; i++){
    for(int j = 0; j < numCols; j++){ 
      printf("%i", h_array[i * numCols + j]);
    }
    printf("\n");
  }

  delete h_array;
  h_array = NULL;

}
/********************/


/** Calculate the mask pixels 
 *
 *  Pixels that have a value of 255 for R, G, and B are mask pixels.
 *  
 *  Outputs an array of 0/1s, where 1s means that pixel should be copied.
 **/
 __global__
void maskKernel(uchar4* d_source, bool* d_mask, size_t numRows, size_t numCols)
{
  //One thread per pixel
  int g_id = threadIdx.x + blockIdx.x * blockDim.x;
  if(g_id > numRows * numCols - 1)
    return;
  uchar4 pix = d_source[g_id];
  d_mask[g_id] = !(pix.x == 255 && pix.y == 255 && pix.z == 255);
}

__global__
void borderPredicateKernel(bool* d_mask,
                           bool* d_border,
                           bool* d_interior,
                           size_t numRows,
                           size_t numCols)
{

  int g_id = threadIdx.x + blockIdx.x * blockDim.x ;
  if(g_id > numRows * numCols - 1)
    return;

  if(d_mask[g_id]) {

    int curCol = threadIdx.x;
    int curRow = blockIdx.x; //Assuming kernel is called with 1 block per row.

    //Calculate neighbors
    int maskedNeighbors = 0;
    //Up
    if(curRow > 0) {
      maskedNeighbors += d_mask[curCol + (curRow - 1) * numCols];
    }
    //Down
    if(curRow < numRows - 1) {
      maskedNeighbors += d_mask[curCol + (curRow + 1) * numCols];
    }
    //Left
    if(curCol > 0) {
      maskedNeighbors += d_mask[curCol - 1 + curRow * numCols];
    }
    //Right
    if(curCol < numCols - 1) {
      maskedNeighbors += d_mask[curCol + 1 + curRow * numCols];
    }

    //Interior if all four neighbors are also in mask
    //Border if in mask, but at least one neighbor is not in mask
    if(maskedNeighbors >= 4) {
      d_interior[g_id] = true;
    } else {
      d_border[g_id] = true;
    }
  }

}

//Taken from problem set 5
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      float* const redChannel,
                      float* const greenChannel,
                      float* const blueChannel)
{
  int g_id = threadIdx.x + blockIdx.x * blockDim.x;

  uchar4 rgba = inputImageRGBA[g_id];
  redChannel[g_id] = (float)rgba.x;
  greenChannel[g_id] = (float)rgba.y;
  blueChannel[g_id] = (float)rgba.z;
}

//Taken from homework 5
__global__
void recombineChannels(const float* const redChannel,
                       const float* const greenChannel,
                       const float* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  int g_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (g_id > numRows * numCols - 1) 
    return;
  outputImageRGBA[g_id].x = (char)redChannel[g_id];
  outputImageRGBA[g_id].y = (char)greenChannel[g_id];
  outputImageRGBA[g_id].z = (char)blueChannel[g_id];

}

__global__
void  jacobiKernel(float* d_in,
                   float* d_out,
                   float* d_sourceChannel,
                   float* d_destChannel,
                   bool* d_border,
                   bool* d_interior,
                   size_t numRows,
                   size_t numCols)
{
  
  int g_id = threadIdx.x + blockIdx.x * blockDim.x;
  if(g_id > numRows * numCols - 1) 
    return;

  if (d_interior[g_id]){
    /*
       1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
          Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

          Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

      2) Calculate the new pixel value:
          float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
          ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]
      */
    float sum1 = 0;
    float sum2 = 0;

    int curCol = threadIdx.x;
    int curRow = blockIdx.x; //Assuming kernel is called with 1 block per row.

    //Lets save this so we only have to access global once instead of (upto) 4 times.
    float sourceValue = d_sourceChannel[g_id];

    //Up
    if(curRow > 0) {
      int neighbor_idx = curCol + (curRow - 1) * numCols;
      if (d_interior[neighbor_idx]) {
        sum1 += d_in[neighbor_idx];
      } else if (d_border[neighbor_idx]) {
        sum1 += d_destChannel[neighbor_idx];
      }      
      sum2 += sourceValue - d_sourceChannel[neighbor_idx];
    }
    //Down
    if(curRow < numRows - 1) {
      int neighbor_idx = curCol + (curRow + 1) * numCols; 
      if (d_interior[neighbor_idx]) {
        sum1 += d_in[neighbor_idx];
      } else if (d_border[neighbor_idx]) {
        sum1 += d_destChannel[neighbor_idx];
      }      
      sum2 += sourceValue - d_sourceChannel[neighbor_idx];
    }
    //Left
    if(curCol > 0) {
      int neighbor_idx = curCol - 1 + curRow * numCols; 
      if (d_interior[neighbor_idx]) {
        sum1 += d_in[neighbor_idx];
      } else if (d_border[neighbor_idx]) {
        sum1 += d_destChannel[neighbor_idx];
      }      
      sum2 += sourceValue - d_sourceChannel[neighbor_idx];
    }
    //Right
    if(curCol < numCols - 1) {
      int neighbor_idx = curCol + 1 + curRow * numCols; 
      if (d_interior[neighbor_idx]) {
        sum1 += d_in[neighbor_idx];
      } else if (d_border[neighbor_idx]) {
        sum1 += d_destChannel[neighbor_idx];
      }      
      sum2 += sourceValue - d_sourceChannel[neighbor_idx];
    }

    float newVal= (sum1 + sum2) / 4.f;
    d_out[g_id] = min(255.f, max(0.f, newVal)); //clamp to [0, 255]

  } else {
    //Not an interior pixel, so just set output to input
    d_out[g_id] = d_destChannel[g_id];
  }
}




void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

    //Allocate device pointers for each image
    uchar4* d_source;
    uchar4* d_dest;
    uchar4* d_blended;  //This is the output image

    int imgSize = numRowsSource * numColsSource * sizeof(uchar4);

    cudaMalloc(&d_source, imgSize);
    cudaMalloc(&d_dest, imgSize);
    cudaMalloc(&d_blended, imgSize);

    //Move images to device memory
    cudaMemcpy(d_source, h_sourceImg, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dest, h_destImg, imgSize, cudaMemcpyHostToDevice);

    //Mask
    bool* d_mask;
    bool* d_borderPred;
    bool* d_interiorPred;
    
    int maskSize = numRowsSource * numColsSource * sizeof(bool);
    
    cudaMalloc(&d_mask, maskSize);
    cudaMalloc(&d_borderPred, maskSize);
    cudaMalloc(&d_interiorPred, maskSize);

    cudaMemset(d_borderPred, 0, maskSize);
    cudaMemset(d_interiorPred, 0, maskSize);
    

  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.
  */

    maskKernel<<<numRowsSource, numColsSource>>>(d_source,
                                                 d_mask,
                                                 numRowsSource,
                                                 numColsSource);
    //This part looks pretty good (10:15am 11/12/2016)
    //printDeviceArray(d_mask, numRowsSource, numColsSource);

  /*
     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.
  */

    borderPredicateKernel<<<numRowsSource, numColsSource>>>(d_mask,
                                                            d_borderPred,
                                                            d_interiorPred,
                                                            numRowsSource,
                                                            numColsSource);
    //printDeviceArray(d_borderPred, numRowsSource, numColsSource);
    //printDeviceArray(d_interiorPred, numRowsSource, numColsSource);
    //This part looks good (10:56am 11/12/2016)

  /*

     3) Separate out the incoming image into three separate channels

  */

    float* d_sourceRed;
    float* d_sourceGreen;
    float* d_sourceBlue;

    float* d_destRed;
    float* d_destGreen;
    float* d_destBlue;

    int channelSize = numRowsSource * numColsSource * sizeof(float);

    checkCudaErrors(cudaMalloc(&d_sourceRed, channelSize));
    checkCudaErrors(cudaMalloc(&d_sourceGreen, channelSize));
    checkCudaErrors(cudaMalloc(&d_sourceBlue, channelSize));

    checkCudaErrors(cudaMalloc(&d_destRed, channelSize));
    checkCudaErrors(cudaMalloc(&d_destGreen, channelSize));
    checkCudaErrors(cudaMalloc(&d_destBlue, channelSize));

    separateChannels<<<numRowsSource, numColsSource>>>(d_source,
                                                       numRowsSource,
                                                       numColsSource,
                                                       d_sourceRed,
                                                       d_sourceGreen,
                                                       d_sourceBlue);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    separateChannels<<<numRowsSource, numColsSource>>>(d_dest,
                                                       numRowsSource,
                                                       numColsSource,
                                                       d_destRed,
                                                       d_destGreen,
                                                       d_destBlue);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  /*
     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.
  */
    float* d_red1;
    float* d_red2;
    float* d_green1;
    float* d_green2;
    float* d_blue1;
    float* d_blue2;

    checkCudaErrors(cudaMalloc(&d_red1, channelSize));
    checkCudaErrors(cudaMalloc(&d_red2, channelSize));
    checkCudaErrors(cudaMalloc(&d_green1, channelSize));
    checkCudaErrors(cudaMalloc(&d_green2, channelSize));
    checkCudaErrors(cudaMalloc(&d_blue1, channelSize));
    checkCudaErrors(cudaMalloc(&d_blue2, channelSize));

    checkCudaErrors(cudaMemcpy(d_red1, d_sourceRed, channelSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_green1, d_sourceGreen, channelSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_blue1, d_sourceBlue, channelSize, cudaMemcpyDeviceToDevice));

  /*
     5) For each color channel perform the Jacobi iteration described 
        above 800 times.
  */

    for(int i = 0; i < 800; i++) {

      jacobiKernel<<<numRowsSource, numColsSource>>>(d_red1,
                                                     d_red2,
                                                     d_sourceRed,
                                                     d_destRed,
                                                     d_borderPred,
                                                     d_interiorPred,
                                                     numRowsSource,
                                                     numColsSource);

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      jacobiKernel<<<numRowsSource, numColsSource>>>(d_green1,
                                                     d_green2,
                                                     d_sourceGreen,
                                                     d_destGreen,
                                                     d_borderPred,
                                                     d_interiorPred,
                                                     numRowsSource,
                                                     numColsSource);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      jacobiKernel<<<numRowsSource, numColsSource>>>(d_blue1,
                                                     d_blue2,
                                                     d_sourceBlue,
                                                     d_destBlue,
                                                     d_borderPred,
                                                     d_interiorPred,
                                                     numRowsSource,
                                                     numColsSource);

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      //Swap buffers 1 and 2 for each color
      float* temp = d_red1;
      d_red1 = d_red2;
      d_red2 = temp;

      temp = d_green1;
      d_green1 = d_green2;
      d_green2 = temp;

      temp = d_blue1;
      d_blue1 = d_blue2;
      d_blue2 = temp;
    }

  /*

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.
  */

    recombineChannels<<<numRowsSource, numColsSource>>>(d_red1,
                                                        d_green1,
                                                        d_blue1,
                                                        d_blended,
                                                        numRowsSource,
                                                        numColsSource);

    cudaMemcpy(h_blendedImg, d_blended, imgSize, cudaMemcpyDeviceToHost);

  /*
      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */

    cudaFree(d_source);
    cudaFree(d_dest);
    cudaFree(d_blended);

    cudaFree(d_mask);
    cudaFree(d_borderPred);
    cudaFree(d_interiorPred);

    cudaFree(d_sourceRed);
    cudaFree(d_sourceGreen);
    cudaFree(d_sourceBlue);
    cudaFree(d_destRed);
    cudaFree(d_destGreen);
    cudaFree(d_destBlue);

    cudaFree(d_red1);
    cudaFree(d_red2);
    cudaFree(d_green1);
    cudaFree(d_green2);
    cudaFree(d_blue1);
    cudaFree(d_blue2);

  /* The reference calculation is provided below, feel free to use it
     for debugging purposes. 
   */

  /*
    uchar4* h_reference = new uchar4[srcSize];
    reference_calc(h_sourceImg, numRowsSource, numColsSource,
                   h_destImg, h_reference);

    checkResultsEps((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * srcSize, 2, .01);
    delete[] h_reference; */
}
