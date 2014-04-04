
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#define DES_BLOCKSIZE 8
#define BLOCKS_PER_LAUNCH 5
#define THREADS_PER_BLOCK 512

void launchKernel(char *h_chunk, int chunkSize, FILE *fp_out);

__global__ void desBlockOperation(char *in, char *out, int n)
{
	//Map the thread id and block id to the DES block
	int i = ((blockIdx.x * THREADS_PER_BLOCK) + threadIdx.x) * DES_BLOCKSIZE;
	if(i < n)
	{
		for (int j = i; j < i + DES_BLOCKSIZE && j < n; j++)
		{
			out[j] = in[j] + j % DES_BLOCKSIZE;
		}
	}
}

int main()
{

	char *fileName = "Text.txt";
	//Open the file for reading
	struct _stat stat;
	_stat(fileName,&stat);
	long fileSize = stat.st_size;
	FILE *fp_in = fopen(fileName , "r");
	char *newFileName = "out.txt";
	FILE *fp_out = fopen(newFileName, "ab");
	char *h_chunkData;
	//Each kernel launch can handle 512 threads each operating on DES_BLOCKSIZE bytes
	///So read the file in DES_BLOCKSIZE * THREADS_PER_BLOCK * BLOCKS_PER_LAUNCH chunks
	if(fp_in && &stat)
	{
		
		int chunkSize = DES_BLOCKSIZE * THREADS_PER_BLOCK * BLOCKS_PER_LAUNCH;
		h_chunkData = (char *)malloc(chunkSize);
		//Read the file in 1 chunk at a time, until fread returns something other than the chunk size
		int lastChunkSize;
		do
		{
			lastChunkSize = fread(h_chunkData, 1, chunkSize, fp_in);
			launchKernel(h_chunkData, lastChunkSize, fp_out);
		}
		while(lastChunkSize == chunkSize);
		fclose(fp_in);
		
	}
    return 0;
}

///Allocate device memory and invoke kernel, writing results to file
void launchKernel(char *h_chunk, int chunkSize, FILE *fp_out)
{
	char *d_chunk = NULL;
	cudaMalloc((void **) &d_chunk, chunkSize);
	cudaMemcpy(d_chunk, h_chunk, chunkSize, cudaMemcpyHostToDevice);
	desBlockOperation<<<BLOCKS_PER_LAUNCH, THREADS_PER_BLOCK>>>(d_chunk, d_chunk, chunkSize);
	cudaMemcpy(h_chunk, d_chunk, chunkSize, cudaMemcpyDeviceToHost);
	fwrite(h_chunk, chunkSize, 1, fp_out);
}

