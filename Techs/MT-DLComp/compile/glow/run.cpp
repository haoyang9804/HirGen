#include "model.h"
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

GLOW_MEM_ALIGN(MODEL_MEM_ALIGN)
uint8_t constantWeight[MODEL_CONSTANT_MEM_SIZE];

/// Statically allocate memory for mutable weights (model input/output data).
GLOW_MEM_ALIGN(MODEL_MEM_ALIGN)
uint8_t mutableWeight[MODEL_MUTABLE_MEM_SIZE];

/// Statically allocate memory for activations (model intermediate results).
GLOW_MEM_ALIGN(MODEL_MEM_ALIGN)
uint8_t activations[MODEL_ACTIVATIONS_MEM_SIZE];

/// Bundle input data absolute address.
uint8_t *inputAddr = GLOW_GET_ADDR(mutableWeight, MODEL_input);

/// Bundle output data absolute address.
uint8_t *outputAddr = GLOW_GET_ADDR(mutableWeight, MODEL_output);


void initConstantWeights(const char *weightsFileName, uint8_t * addr) {
  // Load weights.
  FILE *weightsFile = fopen(weightsFileName, "rb");
  if (!weightsFile) {
    fprintf(stderr, "Could not open the weights file: %s\n", weightsFileName);
    exit(1);
  }
  fseek(weightsFile, 0, SEEK_END);
  size_t fileSize = ftell(weightsFile);
  fseek(weightsFile, 0, SEEK_SET);

  printf("Allocated weights of size: %lu\n", fileSize);

  int result = fread(addr, fileSize, 1, weightsFile);
  if (result != 1) {
    perror("Could not read the weights file");
  } else {
    printf("Loaded weights of size: %lu from the file %s\n", fileSize,
           weightsFileName);
  }
  fclose(weightsFile);
}

void writeOutput(const char *outFileName, uint8_t* addr, int num_bytes) {
  FILE *outFile = fopen(outFileName, "wb");
  if (!outFile) {
    fprintf(stderr, "Could not open the output file: %s\n", outFileName);
    exit(1);
  }

  // int result = fread(addr, fileSize, 1, outFile);
  int result = fwrite(addr, num_bytes, 1, outFile);
  if (result != 1) {
    perror("Could not write output file");
  } else {
    printf("Written successfully to output file\n");
  }
  fclose(outFile);
}

void printArray(uint8_t * addr, int num_elem) {
    float * it = (float*) addr;
    int i;
    for (i = 0; i < num_elem - 1; i++) {
        printf("%f, ", it[i]);
    }
    printf("%f\n", it[num_elem - 1]);
}

void run_model() {
  int errCode = model(constantWeight, mutableWeight, activations);
    if (errCode != GLOW_SUCCESS) {
        printf("Error running bundle: error code %d\n", errCode);
    }
}

double cal_time(int run_times) {
  int i;
  struct timeval start, end;
  run_model();
  gettimeofday(&start, NULL);
  for (i = 0; i < run_times; i++)
    run_model();
  gettimeofday(&end, NULL);
  long timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
  return ((double) timeuse / (double) run_times) / 1000.0;
}

void writeTime(double usedTime, const char* timeFileName) {
  FILE *timeFile = fopen(timeFileName, "w");
  if (!timeFile) {
    fprintf(stderr, "Could not open the time file: %s\n", timeFileName);
    exit(1);
  }

  int result = fprintf(timeFile, "%f\n", usedTime);
  fclose(timeFile);
}

int main(int argc, char ** argv) {
    initConstantWeights("model.weights.bin", constantWeight);
    initConstantWeights(argv[1], inputAddr);

    if (argc > 2) {
        double runTime = cal_time(5);
        writeTime(runTime, "time.bin");
    }
    else {
        run_model();
    }

    printArray(outputAddr, 40);
    writeOutput("out.bin", outputAddr, 160);
}
