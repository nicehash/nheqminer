//compile with
//gcc -o solver solver.c equihash_avx2.o
//
//./solver
//sha256sum out2.bin
//Expected result with default input.bin (beta1 testnet block 2),
//257d3c3250c14978614ac169edcf72bd131a2e4c227c8d7e21a2cd6131a13dda  out2.bin

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>	//for rdtsc

#define CONTEXT_SIZE 178033152

//Linkage with assembly
//EhPrepare takes in 136 bytes of input. The remaining 4 bytes of input is fed as nonce to EhSolver.
//EhPrepare saves the 136 bytes in context, and EhSolver can be called repeatedly with different nonce.
void EhPrepare(void *context, void *input);
int32_t EhSolver(void *context, uint32_t nonce);
extern char testinput[];

//context is the memory used for Equihash computation. It should be allocated outside of SolverFunction, the size is defined by CONTEXT_SIZE, about 180MB.
//SolverFunction API has slight overhead in mining due to missing opportunity to run EhSolver multiple times after a single EhPrepare.
int SolverFunction(void* context, const unsigned char* input,
	bool (*validBlock)(void*, const unsigned char*),
	void* validBlockData,
	bool (*cancelled)(void*),
	void* cancelledData,
	int numThreads,
	int n, int k)
{
	int numsolutions, i;

	EhPrepare(context, (void *) input);
	numsolutions = EhSolver(context, *(uint32_t *)(input+136));

	for (i=0; i<numsolutions; i++) {
		validBlock(validBlockData, (unsigned char*)(context+1344*i));
	}
	return numsolutions;
}

bool validBlock(void *validBlockData, const unsigned char *solution)
{
	return 0;
}

bool cancelled(void *cancelledData)
{
	return 0;
}

int main(void)
{
	void *context_alloc, *context, *context_end;
	uint32_t *pu32;
	uint64_t *pu64, previous_rdtsc;
	uint8_t inputheader[144];	//140 byte header
	FILE *infile, *outfile;
	struct timespec time0, time1;
	uint64_t rdtsc0, rdtsc1;
	long t0, t1;
	int32_t numsolutions;
	int i, j;
	char outfilename[32];

	context_alloc = malloc(CONTEXT_SIZE+4096);
	context = (void*) (((long) context_alloc+4095) & -4096);
	context_end = context + CONTEXT_SIZE;

	//Init page tables. This is not necessary, but useful to get a more consistent single-run timing.
	for (pu32=context; (void*) pu32<context_end; pu32+=1024)
		*pu32 = 0;

	infile = 0;
	infile = fopen("input.bin", "rb");
	if (infile) {
		puts("Reading input.bin");
		fread(inputheader, 140, 1, infile);
		fclose(infile);
	} else {
		puts("input.bin not found, use sample data (beta1 testnet block 2)");
		memcpy(inputheader, testinput, 140);
	}

	puts("Running solver...");
	clock_gettime(CLOCK_MONOTONIC, &time0);
	rdtsc0 = __rdtsc();
	numsolutions = SolverFunction(context, inputheader, validBlock, 0, cancelled, 0, 1, 200, 9);
	//EhPrepare(context, (void *) inputheader);
	//numsolutions = EhSolver(context, *(uint32_t *)(inputheader+136));
	clock_gettime(CLOCK_MONOTONIC, &time1);
	rdtsc1 = __rdtsc();

	//Print some debug information
	pu64 = (uint64_t *) (context + 102408);	//Read the debug area for statistics
	printf("BLAKE2b rdtsc: %lu\n", pu64[1]-pu64[0]);
	previous_rdtsc = pu64[1];
	for (i=1, j=2; i<=9; i++, j+=2) {
		printf("Stage %u, Output pairs %u, rdtsc: %lu\n", i, (uint32_t) pu64[j+1], pu64[j]-previous_rdtsc);
		previous_rdtsc = pu64[j];
	}
	printf("Number of solutions before duplicate removal: %u\n", *(uint32_t *) (context+16384));
	printf("Duplicate removal and tree expand rdtsc: %lu\n", pu64[j]-previous_rdtsc);

	printf("Number of solutions: %d\n", numsolutions);

	j = numsolutions < 4 ? numsolutions : 4;
	for (i=0; i<j; i++) {
		sprintf(outfilename, "out%d.bin", i);
		outfile = fopen(outfilename, "wb");
		fwrite(context+1344*i, 1344, 1, outfile);
		fclose(outfile);
	}

	t0 = time0.tv_sec * 1000000000 + time0.tv_nsec;
	t1 = time1.tv_sec * 1000000000 + time1.tv_nsec;
	printf("Time: %ld ms\n", (t1-t0)/1000000);
	t0 = (t1-t0)/1000;
	printf("Measure rdtsc frequency = %.3f MHz\n", (double) (rdtsc1-rdtsc0)/t0);

	free(context_alloc);
	return 0;
}
