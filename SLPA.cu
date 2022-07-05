#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <curand_kernel.h>
#include <curand.h>
#include <stdio.h>


#define N 1
#define B 1


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct triples {
    int key;
    int val;
    int iter;
}triples;

//update labellist with random label
__global__ void SLPASpeaker(int * row, int *col, int *val, int * memnnz, triples *mem, triples *labellist, int n, int T, curandState *states, uint64_t seed) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, tid, 0, &states[tid]);
    for(int i = tid; i < row[n]; i+=N*B) { 
        int offset = ((int)curand_uniform(&states[tid]) %(memnnz[i]+1)); 
        labellist[i] = mem[col[i]*n + offset];
        labellist[i].val = 1;
        labellist[i].iter = T;
    }
}

//get frequency of labels 
__global__ void SLPAListener1(int * row, int *col, int *val, triples *labellist, int * row_id, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //get value from labellist
    for(int i = tid; i < row[n]; i+=N*B) {
        //get offsets of specific lists
        int offset1 = row[row_id[i]];
        int offset2 = row[row_id[i+1]];
        __shared__ int count[N];
        count[i%N] = 0;
        __shared__ int startval[N];
        startval[i%N] = labellist[i].key;
        //go over list and count how often that label exists in the list
        for(int j = offset1; j < offset2; j++) {
            if(startval[i%N] == labellist[j].key)
                count[i%N]++; 
        }
        labellist[i].val = count[i%N];
    }
}
//get most seen label 
__global__ void SLPAListener2(int * row, int *col, int *val, triples *labellist, int n) {
    for(int i = blockIdx.x; i < n; i+=blockDim.x) {
        int offset1 = row[i];
        int offset2 = row[i+1];
        int x = offset2-offset1;
        while(threadIdx.x < x/2) {
            labellist[threadIdx.x] = (labellist[threadIdx.x].val > labellist[threadIdx.x+(x/2)].val) ? labellist[threadIdx.x] : 
                labellist[threadIdx.x+(x/2)];
            x = x/2;
            __syncthreads();
        }
    }
}
//update listener per node memory
__global__ void SLPAListener3(int * row, int *col, int *val, int * memnnz, triples *mem, triples *labellist, int n, int T) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = tid; i < n; i+= N*B) {
        if(memnnz[i] < n) { //wiil be d+1 in the future 
            int flag = 0;
            for(int j = 0; j < n; j++) {
                if(mem[n*i+j].key == labellist[row[i]].key) {
                    mem[n*i+j].val++;
                    flag = 1;
                }
            }
            if(flag == 0) {
                mem[n*i+memnnz[i]] = labellist[row[i]];
                mem[n*i+memnnz[i]].val = 1;
                memnnz[i]++;
            }
        }
        else{
            int lowcount = 1000000000;
            int index = -1;
            for(int j = 0; j < n; j++) {
                if(mem[n*i+j].val < lowcount) {
                    index = j;
                    lowcount = mem[n*i+j].val;
                }
            }
            mem[n*i+index] = labellist[row[i]];
            mem[n*i+index].val = 1; 
        } 
    }
}

__global__ void SLPAPostProcess(triples *mem, int n, int T, int r) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = tid; i < n*n; i++) { //d+1
        if(mem[i].val < T*r) {
            mem[i].key = -1;
            mem[i].val = 0;
        }
    }
}

// __global__ void SLPA(int *row, int *col, int *val, int * mem, int n, int T, float r, curandState *states) {
//     int tid = blockDim.x * blockIdx.x + threadIdx.x;
//     curand_init(seed, tid, 0, &states[tid]);
//     for(int t = 0; t < T; t++) {
//         for(int i = tid; i < n; i++) {
//             int labellist[n] = {0};
//             for(int j = row[i]; j < row[i+1]; j++) {
//                 labellist[col[j]] = mem[col[j]*n + (curand_uniform(&states[id]) %(n+1))];
//             }
//             int pick = labellist[0];
//             for(int j = 0; j < n; j++) {
                
//             }
//         }
//     }
// }


int main() {
    /* 0 1 0
       1 0 0
       1 1 0

    */
    int n = 3;
    int * row;
    int *col, *val, *row_id, *memnnz;
    triples *mem, *labellist;
    std::cout << "malloc call\n";
    gpuErrchk(cudaMallocManaged(&row, 4 * sizeof(int)));
    // cudaMallocManaged(&col, 4 * sizeof(int));
    // cudaMallocManaged(&val, 4 * sizeof(int));
    // cudaMallocManaged(&memnnz, 3 * sizeof(int));
    // cudaMallocManaged(&row_id, 3 * sizeof(int));
    // cudaMallocManaged(&mem, n*n*sizeof(triples));
    // cudaMallocManaged(&labellist, n*sizeof(triples));
    
    row[0] = 0;
    std::cout << row[0] << "\n";
    exit(0);
    row[1] = 1;
    row[2] = 2;
    row[3] = 4;

    col[0] = 1;
    col[1] = 0;
    col[2] = 1;
    col[3] = 1;

    val[0] = 1;
    val[1] = 1;
    val[2] = 1;
    val[3] = 1;

    row_id[0] = 0;
    row_id[1] = 1;
    row_id[2] = 1;
    row_id[3] = 2;


    for(int i = 0; i < n; i++) {
        labellist[i].key = -1;
        labellist[i].val = 0;
        memnnz[i] = 1;
        for(int j = 0; j < n; j++) {
            mem[i*n + j].key = i;
            mem[i*n + j].val = 1;
        }
    }
    int T = 5;
    float r = 0.3;
    curandState *dev_random;
    cudaMallocManaged(&dev_random, N*B*sizeof(curandState));

    for(int i = 1; i < T; i++) {
        //speaker rule
        SLPASpeaker<<<B,N>>>(row,col,val, memnnz, mem, labellist, n, T, dev_random, time(NULL));
        cudaDeviceSynchronize();
        //label frequency
        SLPAListener1<<<B,N>>>(row,col,val, labellist, row_id, n);
        cudaDeviceSynchronize();
        //max frequency
        SLPAListener2<<<B,N>>>(row,col,val, labellist, n);
        cudaDeviceSynchronize();
        //memory update
        SLPAListener3<<<B,N>>>(row,col,val, memnnz, mem, labellist, n, T);
        cudaDeviceSynchronize();
    }
    SLPAPostProcess<<<B,N>>>(mem, n, T, r);
    cudaDeviceSynchronize();
    //SLPA<<<B,N>>>(row,col,val,mem, n, T,r, dev_random);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(mem[i*n+j].key != -1) {
                std::cout << mem[i*n+j].key << " ";
            }
            std::cout << "overlap\n";
        }
    }
    std::cout << "\n";
    return 0;
}