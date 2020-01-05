// Multiply two matrices A * B = C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <oclUtils.h>
#include "CL/cl.h"

#define WA 1024
#define HA 1024
#define WB 1024
#define HB 1024
#define WC 1024
#define HC 1024
#define MAX_SOURCE_SIZE 0x100000
#define print 1	// Set this to 1 if you want to see matrix details


/////////////////////////////////////////////////////////
// Allocates a matrix with random float entries.
/////////////////////////////////////////////////////////
void randomInit(float* data, int size)
{
	int i;
	for (i = 0; i < size; ++i) {
		data[i] = rand() / (float) RAND_MAX;
	}
}
/////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
	int i;
	// set seed for rand()
	srand(2006);

	// 1. allocate host memory for matrices A and B
	unsigned int size_A = WA * HA;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A = (float*) malloc(mem_size_A);

	unsigned int size_B = WB * HB;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float* h_B = (float*) malloc(mem_size_B);

	// 2. initialize host memory
	printf("Loading Matrix A and B with random numbers\n %");
	randomInit(h_A, size_A);
	randomInit(h_B, size_B);

	// 3. print out A and B
#if print
	printf("\n\nMatrix A\n");
	for(i = 0; i < size_A; i++)
	{
		printf("%lf ", h_A[i]);
		if(((i + 1) % WA) == 0)
			printf("\n");
	}

	printf("\n\nMatrix B\n");
	for(i = 0; i < size_B; i++)
	{
		printf("%lf ", h_B[i]);
		if(((i + 1) % WB) == 0)
			printf("\n");
	}
	printf("\n\n");
#endif

	// 4. allocate host memory for the result C
	unsigned int size_C = WC * HC;
	unsigned int mem_size_C = sizeof(float) * size_C;
	float* h_C = (float*) malloc(mem_size_C);

	// 5. Initialize OpenCL
	// OpenCL specific variables
	cl_context clGPUContext;
	cl_command_queue clCommandQue;
	cl_program clProgram;
	cl_kernel clKernel;

	size_t dataBytes;
	size_t kernelLength;
	cl_int errcode;

	// OpenCL device memory for matrices
	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;

	char* info;
	cl_uint count = 0;
	cl_uint ret_num_devices;
	size_t infoSize;
	cl_device_id device_id = NULL;
	cl_platform_id *platform = NULL;
	cl_platform_id platform_id = NULL;
	cl_platform_id platform_id_amd = NULL;

	clGetPlatformIDs(10, NULL, &count);
	platform = (cl_platform_id*) malloc(sizeof(cl_platform_id) * count);
	clGetPlatformIDs(count, platform, NULL);
	if (count == 0) {
		printf("iC-PHYS create_thread Error: no platform found. Shutting down iCPHYS.\n");
		exit(1);
	}
	printf("fetching devices..\n");
	int device_found = 0;
	for (i = 0; i < count; i++) {
		clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 0, NULL, &infoSize);
		info = (char*) malloc(infoSize);
		clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, infoSize, info, NULL);
//		printf("platform info : %s\n",info);
		if(strcmp(info,"AMD Accelerated Parallel Processing") != 0 ) {
			platform_id = platform[i];
			device_found++;
			free(info);
		} else if(strcmp(info,"AMD Accelerated Parallel Processing") == 0 ) {
			platform_id_amd = platform[i];
			device_found++;
			free(info);
		}
	}
//	if (device_found < 2) {
//		printf("iC-PHYS Error create_thread: AMD Accelerated Parallel Processing Device/GPU Device not found. Shutting down iCPHYS.\n");
//		exit(1);
//	}
	errcode = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
	if (errcode != 0) {
		printf("iC-PHYS Error: clGetDeviceIDs error: %d. couldn't load\n", errcode);
		exit(1);
	}
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &infoSize);
	info = (char*) malloc(infoSize);
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, infoSize, info, NULL);
	printf("Device selected: %s\n",  info);
	clGPUContext = clCreateContext(NULL, 1, &device_id, NULL, NULL, &errcode);
	if (errcode != 0) {
		printf( "iC-PHYS Error: clCreateContext error: %d. couldn't load\n", errcode);
		exit(1);
	}


	//Create a command-queue
	clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
	// shrCheckError(errcode, CL_SUCCESS);
	if (errcode != 0) {
		printf("iC-PHYS Error: clCreateCommandQueue error: %d. couldn't load\n", errcode);
		exit(1);
	}


	// Setup device memory
	d_C = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, mem_size_C, NULL, &errcode);
	d_A = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, h_A, &errcode);
	d_B = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_B, h_B, &errcode);


	// 6. Load and build OpenCL kernel
	// Reading opencl kernel code file
	FILE *fp;
	char *fileName = "src/matmul_kernel.cl";
	char *source_str;
	size_t source_size;

	/* Load the source code containing the kernel*/
	fp = fopen(fileName, "r");
	if (!fp) {
		printf("Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*) malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, &source_size, &errcode);
	// shrCheckError(errcode, CL_SUCCESS);
	if (errcode != 0) {
		printf("iC-PHYS Error: clCreateProgramWithSource error: %d. couldn't load\n", errcode);
		exit(1);
	}

	printf("Building Program\n");
	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	// errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	// shrCheckError(errcode, CL_SUCCESS);

	if (errcode != 0) {
		printf("iC-PHYS Error: clBuildProgram error: %d. couldn't load\n", errcode);
		size_t log_size;
		clGetProgramBuildInfo(clProgram, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char *log = (char *) malloc(log_size);
		clGetProgramBuildInfo(clProgram, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("%s\n", log);
		exit(1);
	}

	clKernel = clCreateKernel(clProgram, "matrixMul", &errcode);

	// shrCheckError(errcode, CL_SUCCESS);
	if (errcode != 0) {
		printf("iC-PHYS Error: clCreateKernel error: %d. couldn't load\n", errcode);
		exit(1);
	}

	// 7. Launch OpenCL kernel
	size_t localWorkSize[2], globalWorkSize[2];

	int wA = WA;
	int wC = WC;
	errcode = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&d_C);
	if (errcode != 0) {
		printf("iC-PHYS Error: clCreateKernel1 error: %d. couldn't load\n", errcode);
		exit(1);
	}
	errcode = clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&d_A);
	if (errcode != 0) {
		printf("iC-PHYS Error: clCreateKernel2 error: %d. couldn't load\n", errcode);
		exit(1);
	}
	errcode = clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&d_B);
	if (errcode != 0) {
		printf("iC-PHYS Error: clCreateKernel3 error: %d. couldn't load\n", errcode);
		exit(1);
	}
	errcode = clSetKernelArg(clKernel, 3, sizeof(int), (void *)&wA);
	if (errcode != 0) {
		printf("iC-PHYS Error: clCreateKernel4 error: %d. couldn't load\n", errcode);
		exit(1);
	}
	errcode = clSetKernelArg(clKernel, 4, sizeof(int), (void *)&wC);
	if (errcode != 0) {
		printf("iC-PHYS Error: clCreateKernel5 error: %d. couldn't load\n", errcode);
		exit(1);
	}

	localWorkSize[0] = 16;
	localWorkSize[1] = 16;
	globalWorkSize[0] = 512;
	globalWorkSize[1] = 512;

	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (errcode != 0) {
		printf("iC-PHYS Error: clEnqueueNDRangeKernel error: %d. couldn't load\n", errcode);
		exit(1);
	}

	// 8. Retrieve result from device
	errcode = clEnqueueReadBuffer(clCommandQue, d_C, CL_TRUE, 0, mem_size_C, h_C, 0, NULL, NULL);
	if (errcode != 0) {
		printf("iC-PHYS Error: clEnqueueReadBuffer error: %d. couldn't load\n", errcode);
		exit(1);
	}


	// 9. print out the results
	printf("Matrix C Results Received\n");
#if print
	for(i = 0; i < size_C; i++)
	{
		printf("%lf ", h_C[i]);
		if(((i + 1) % WC) == 0)
			printf("\n");
	}
	printf("\n");
#endif

	// 10. clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	clReleaseMemObject(d_A);
	clReleaseMemObject(d_C);
	clReleaseMemObject(d_B);

	// free(device_id);
	// free(clMatrixMul);
	clReleaseContext(clGPUContext);
	clReleaseKernel(clKernel);
	clReleaseProgram(clProgram);
	clReleaseCommandQueue(clCommandQue);
	return 0;

}
