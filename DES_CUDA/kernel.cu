
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#define DES_BLOCKSIZE 8

__global__ void desBlockOperation(char *in, char *out, int n)
{
	int i = threadIdx.x * DES_BLOCKSIZE;
	if(i < n)
	{
		for (int j = i; j < i+DES_BLOCKSIZE && j < n; j++)
		{
			out[j] = in[j] + j%DES_BLOCKSIZE;
		}
	}
}

int main()
{
	char *fileName = "Text.txt";
	//Open the file for reading
	struct _stat stat;
	_stat(fileName,&stat);
	int fileSize = stat.st_size;
	FILE *fp_in = fopen(fileName , "r");
	char *h_text;
	//TODO: Read the file in multiple chunks
	if(fp_in && &stat)
	{
		h_text = (char *)malloc(fileSize);
		char *s = (char *)malloc(sizeof(char *));
		fread(h_text, fileSize, 1, fp_in);
		fclose(fp_in);
	}
	if(h_text){
		//Allocate memory on the device
		char *d_text = NULL;
		cudaMalloc((void **) &d_text, fileSize);
		//Copy input data to device
		cudaMemcpy(d_text, h_text, fileSize, cudaMemcpyHostToDevice);
		//Invoke kernel
		int threadsPerBlock = (fileSize + DES_BLOCKSIZE -1) / DES_BLOCKSIZE;
		desBlockOperation<<<1, threadsPerBlock>>>(d_text, d_text, fileSize);
		//Copy data from device to host
		cudaMemcpy(h_text, d_text, fileSize, cudaMemcpyDeviceToHost);
		
		//Write the output to a file
		char *newFileName = "out.txt";
		FILE *fp_out = fopen(newFileName, "w");
		if(fp_out)
		{
			fwrite(h_text, fileSize, 1, fp_out);
		}
	}
    return 0;
}


