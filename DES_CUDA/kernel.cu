
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DES_BLOCKSIZE 8

__global__ void desBlockOperation(char *in, int n)
{
	int i = threadIdx.x;
	if (i < n)
	{
		in[i] = in[i]+1;
	}
}

int main()
{
	char *fileName = "Text.txt";
	//Open the file for reading
	struct _stat stat;
	_stat(fileName,&stat);
	int fileSize = stat.st_size;
	FILE *fp = fopen(fileName , "r");
	//TODO: Use as many blocks as neccessary
	//Right now, there's just one(fine for files under 4KB)
	char **thisBlock;
	if(fp && &stat)
	{
		thisBlock = (char **)malloc(512 * sizeof(char *));
		
		//Each thread is 64 bits = 8 bytes
		//Each block is up to 512 threads
		char s[DES_BLOCKSIZE];
		int i = 0;
		while ((fgets(s,DES_BLOCKSIZE,fp)) != NULL)
		{
			thisBlock[i] = (char *)malloc(sizeof(char *));
			memcpy(thisBlock[i], s, DES_BLOCKSIZE);
			i++;
		}
		fclose(fp);
	}
	if(thisBlock){
		//Flatten the array
		char *thisBlock_flat = &(thisBlock[0][0]);
		//Allocate device memory
		char *d_thisBlock = NULL;
		char *result = (char *)malloc(512);
		int size = 512 * sizeof(char *);
		cudaMalloc(&d_thisBlock, size * DES_BLOCKSIZE);
		cudaMemcpy(d_thisBlock, thisBlock_flat, size * DES_BLOCKSIZE, cudaMemcpyHostToDevice);
		//number of items is filesize in bytes / 8
		int n = (fileSize + DES_BLOCKSIZE - 1) / DES_BLOCKSIZE; 
		//Invoke the kernel on this thread block
		desBlockOperation<<<1, 512>>>(d_thisBlock, n);
		cudaMemcpy(result, d_thisBlock, size, cudaMemcpyDeviceToHost);
		int i = 0;
	}
    return 0;
}


