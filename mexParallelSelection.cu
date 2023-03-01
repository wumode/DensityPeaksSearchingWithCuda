//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include "mex.h"
#include "gpu\mxGPUArray.h"

//__constant__ int index_offset3x3[9];


__global__ void computeDeltaKernel(int* rhoMat, int* pxMat, int* pyMat, float* delta, int* signMatU, int* index_offset3x3, int height, int width, int d_height)
{
    int idx = blockIdx.x *blockDim.x + threadIdx.x;
    int x = idx / height;
    int y = idx % height;
    if (0<x&&x<width-1 &&0<y&&y<height-1) {
        //int rho = rhoMat[idx];
        int rhoVector[] = { rhoMat[idx - height - 1], rhoMat[idx - height], rhoMat[idx - height + 1],
                            rhoMat[idx - 1],          rhoMat[idx],          rhoMat[idx + 1],
                            rhoMat[idx + height - 1], rhoMat[idx + height], rhoMat[idx + height + 1] };
        int rho_max = 0;
        unsigned char max_index = 0;
        for (int i = 0; i < 4; ++i) {
            if (rhoVector[i] >= rho_max) {
                rho_max = rhoVector[i];
                max_index = i;
            }
        }
        for (int i = 4; i < 9; ++i) {
            if (rhoVector[i] > rho_max) {
                rho_max = rhoVector[i];
                max_index = i;
            }
        }
        if (max_index == 4) {
            signMatU[idx] = 1;
        }
        else {
            signMatU[idx] = 0;
            int idmax = idx + index_offset3x3[max_index];
            float dist = (float)((pxMat[idmax] - pxMat[idx]) * (pxMat[idmax] - pxMat[idx]) + 
                (pyMat[idmax] - pyMat[idx]) * (pyMat[idmax] - pyMat[idx]));
            dist = sqrt(dist);
            int d_idx = d_height * (pxMat[idx]-1) + pyMat[idx]-1;
            delta[d_idx] = dist;
        }
    }
}


cudaError_t parallelSelectionWithCuda(int* rhoMat, int* pxMat, int* pyMat, float* delta, int* signMatU, int height, int width, int d_height, int d_width) {
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        mexPrintf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
    int* dev_rhoMat = 0;
    int* dev_pxMat = 0;
    int* dev_pyMat = 0;
    float* dev_delta = 0;
    int* dev_signMatU = 0;
    int* dev_index_offset3x3 = 0;
    unsigned int size =height*width ;
    unsigned int dsize = d_height*d_width ;
    int index_offset3x3[] = { -height - 1, -height, -height + 1, -1, 0, 1, height - 1, height, height + 1 };
    //cudaMemcpyToSymbol(index_offset3x3, index_offset3x3_tmp, sizeof(int) * 9);

    cudaStatus = cudaMalloc((void**)&dev_index_offset3x3, 9 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        mexPrintf( "cudaMalloc dev_index_offset3x3 %d bytes failed!\n", sizeof(int) * 9 );
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_rhoMat, size* sizeof(int));
    if (cudaStatus != cudaSuccess) {
        mexPrintf("cudaMalloc dev_rhoMat %d bytes failed!\n", size * sizeof(int));
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_pxMat, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        mexPrintf( "cudaMalloc dev_pxMat failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_pyMat, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        mexPrintf( "cudaMalloc dev_pyMat failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_delta, dsize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        mexPrintf( "cudaMalloc dev_delta failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_signMatU, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        mexPrintf( "cudaMalloc dev_signMatU failed!");
        goto Error;
    }

    dim3 blockSize(256);
    dim3 gridSize((int)((size + blockSize.x - 1) / blockSize.x));

    //mexPrintf("gridSize: %d\nblockSize: %d\n", gridSize.x, blockSize.x);
    // Copy input vectors from host memory to GPU buffers.

    cudaStatus = cudaMemcpy(dev_index_offset3x3, index_offset3x3, 9 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        mexPrintf("cudaMemcpy dev_index_offset3x3 failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_rhoMat, rhoMat, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        mexPrintf("cudaMemcpy dev_rhoMat failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_pxMat, pxMat, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        mexPrintf("cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_pyMat, pyMat, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        mexPrintf("cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_delta, delta, dsize * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        mexPrintf("cudaMemcpy failed!");
        goto Error;
    }
    computeDeltaKernel << <gridSize, blockSize >> > (dev_rhoMat, dev_pxMat, dev_pyMat, dev_delta, dev_signMatU, dev_index_offset3x3, height, width, d_height);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        mexPrintf( "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        mexPrintf( "cudaDeviceSynchronize returned error code %d after launching computeDeltaKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(signMatU, dev_signMatU, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        mexPrintf( "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(delta, dev_delta, dsize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        mexPrintf( "cudaMemcpy failed!");
        goto Error;
    }
    return cudaStatus;
Error:
    cudaFree(dev_rhoMat);
    cudaFree(dev_pxMat);
    cudaFree(dev_pyMat);
    cudaFree(dev_delta);
    cudaFree(dev_signMatU);
    cudaFree(dev_index_offset3x3);
    return cudaStatus;
}


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, mxArray const *prhs[])
{
    if (nrhs != 5) {
        mexErrMsgTxt("Wrong number of input arguments.\n");
    }
    if (nlhs > 5) {
        mexErrMsgTxt("Too many output argumnents.\n");
    }

    int* rhoMat = (int*)mxGetPr(prhs[0]);
    int* pxMat = (int*)mxGetPr(prhs[1]);
    int* pyMat = (int*)mxGetPr(prhs[2]);
    float* delta = (float*)mxGetPr(prhs[3]);

    int M = mxGetM(prhs[0]);
    int N = mxGetN(prhs[0]);
    int* sign = (int*)malloc(sizeof(int) * M * N);

    int dM = mxGetM(prhs[3]);
    int dN = mxGetN(prhs[3]);
    //mexPrintf("%dx%d\n", dM, dN);
    //mexPrintf("%dx%d\n", M, N);
    cudaError_t cudaStatus = parallelSelectionWithCuda(rhoMat, pxMat, pyMat, delta, sign, M, N, dM, dN);
    if (cudaStatus != cudaSuccess) {
        mexErrMsgTxt("Exit with CUDA error.\n");
    }

    plhs[0] = mxCreateDoubleMatrix(M, N, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(dM, dN, mxREAL);

    double* sign_out = mxGetPr(plhs[0]);
    double* delta_out = mxGetPr(plhs[1]);


    for (int i = 0; i < dM*dN; ++i) {
        delta_out[i] = delta[i];
    }
    for (int i = 0; i < M*N; ++i) {
        sign_out[i] = sign[i];
    }
    free(sign);
}