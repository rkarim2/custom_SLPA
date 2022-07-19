#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <curand_kernel.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <fstream> 

#define DEBUGS 1
#define DEBUG1 1
#define DEBUG2 1
#define DEBUG3 1
#define N 256
#define B 5


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

// //update labellist with random label
// __global__ void SLPASpeaker(int * row, int *col, int *val, int * memnnz, int * row_id, triples *mem, triples *labellist, int n, int T, curandState *states, uint64_t seed) {
//     int tid = blockDim.x * blockIdx.x + threadIdx.x;
//     curand_init(seed, tid, 0, &states[tid]);
//     for(int i = tid; i < row[n]; i+=N*B) { 
//         float rand = curand_uniform(&states[tid]);
//         float rand2 = rand * memnnz[col[i]];
//         int offset = (int)truncf(rand2);
//         // if(49 <= i && i < 56) {
//         // printf("memnnz is %d\n", memnnz[col[i]]);
//         // printf("original rand %f and new rand %f\n", rand, rand2);
//         // printf("The offset for %d is %d\n", i, offset);
//         // }
//         //int offset = ((int)curand_uniform(&states[tid]) %(memnnz[i]+1)); 
//         labellist[i] = mem[col[i]*n + offset];
//         labellist[i].val = 1;
//         labellist[i].iter = T;
//     }
// }

//no random speaker
__global__ void SLPASpeaker(int * row, int *col, int *val, int * memnnz, int * row_id, triples *mem, triples *labellist, int n, int T, curandState *states, uint64_t seed) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //curand_init(seed, tid, 0, &states[tid]);
    for(int i = tid; i < row[n]; i+=N*B) { 
        int max = 0;
        int offset = -1;
        for(int j = 0; j < memnnz[col[i]]; j++) {
            if(mem[col[i]*n +j].val > max) {
                offset = j;
                max = mem[col[i]*n +j].val; 
            }
        }
        labellist[i] = mem[col[i]*n + offset];
        labellist[i].val = 1;
        labellist[i].iter = T;
    }
}



//update labellist with label that is proportional to random value
// __global__ void SLPASpeaker(int * row, int *col, int *val, int * memnnz, int * row_id, triples *mem, triples *labellist, int n, int T, curandState *states, uint64_t seed) {
//     int tid = blockDim.x * blockIdx.x + threadIdx.x;
//     __shared__ float rand;
//     if(threadIdx.x == 0) {
//         curand_init(seed, tid, 0, &states[tid]);
//         rand = curand_uniform(&states[tid]);
//     }
//     __syncthreads();
//     for(int i = tid; i < row[n]; i+=N*B) { 
//         int flag = 0;
//         for(int j = 0; j < memnnz[col[i]]; j++) {
//             if( mem[col[i]*n+j].val <= (int)(rand*T)+1 && mem[col[i]*n+j].val >= (int)(rand*T)-1 && flag == 0) {
//                 labellist[i] = mem[col[i]*n + j];
//                 labellist[i].val = 1;
//                 labellist[i].iter = T;
//                 flag = 1;
//             }
//         }
//         // if(49 <= i && i < 56) {
//         // printf("memnnz is %d\n", memnnz[col[i]]);
//         // printf("original rand %f and new rand %f\n", rand, rand2);
//         // printf("The offset for %d is %d\n", i, offset);
//         // }
//         //int offset = ((int)curand_uniform(&states[tid]) %(memnnz[i]+1)); 
//         if(flag == 0) {
//         float rand2 = rand * memnnz[col[i]];
//         int offset = (int)truncf(rand2);
//         labellist[i] = mem[col[i]*n + offset];
//         labellist[i].val = 1;
//         labellist[i].iter = T;
//         }
//     }
// }


//get frequency of labels 
__global__ void SLPAListener1(int * row, int *col, int *val, triples *labellist, int * row_id, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //get value from labellist
    for(int i = tid; i < row[n]; i+=N*B) {
        //get offsets of specific lists
        int offset1 = row[row_id[i]];
        int offset2 = row[row_id[i]+1];
        __shared__ int count[N];
        count[tid%N] = 0;
        //go over list and count how often that label exists in the list
        for(int j = offset1; j < offset2; j++) {
            if(labellist[i].key == labellist[j].key)
                count[tid%N]++; 
        }
        labellist[i].val = count[tid%N];
    }
}

//see if a tiebreak is needed at this point 

//FIXME need to do warp reduction
//get most seen label 
__global__ void SLPAListener2(int * row, int *col, int *val, triples *labellist, int n, int * row_id) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = tid; i < row[n]; i+=N*B) {
        if(labellist[row[row_id[i]]].val < labellist[i].val) {
            atomicCAS(&(labellist[row[row_id[i]]].key), labellist[row[row_id[i]]].key, labellist[i].key);
            atomicCAS(&(labellist[row[row_id[i]]].val), labellist[row[row_id[i]]].val, labellist[i].val);
            atomicCAS(&(labellist[row[row_id[i]]].iter), labellist[row[row_id[i]]].iter, labellist[i].iter);
        }
    }
}

//see if a tie needs to be broken
__global__ void SLPAListener2_1(int * row, triples *labellist, int n, int * row_id, int * labellistnnz) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = tid; i < row[n]; i+=N*B) {
        if(labellist[row[row_id[i]]].val == labellist[i].val && labellist[i].key != labellist[row[row_id[i]]].key) {
            atomicCAS(&(labellist[row[row_id[i]]+labellistnnz[row_id[i]]].key), labellist[row[row_id[i]]+labellistnnz[row_id[i]]].key, labellist[i].key);
            atomicCAS(&(labellist[row[row_id[i]]+labellistnnz[row_id[i]]].val), labellist[row[row_id[i]]+labellistnnz[row_id[i]]].val, labellist[i].val);
            atomicCAS(&(labellist[row[row_id[i]]+labellistnnz[row_id[i]]].iter), labellist[row[row_id[i]]+labellistnnz[row_id[i]]].iter, labellist[i].iter);
            atomicAdd(&(labellistnnz[row_id[i]]), 1);
        }
    }
}

//break tie
__global__ void SLPAListener2_2(int * row, triples *labellist, int n, int * row_id, int * labellistnnz, curandState *states, uint64_t seed) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, tid, 0, &states[tid]);
    for(int i = tid; i < n; i=N*B) {
        float rand = curand_uniform(&states[tid]);
        float rand2 = rand * labellistnnz[i];
        int offset = (int)truncf(rand2);
        labellist[row[i]] = labellist[row[i]+offset];
    }
}


// //get most seen label 
// __global__ void SLPAListener2(int * row, int *col, int *val, triples *labellist, int n) {
//     for(int i = blockIdx.x; i < n; i+=blockDim.x) {
//         int offset1 = row[i];
//         int offset2 = row[i+1];
//         int x = offset2-offset1;
//         if(i == 12) {
//             printf("%d\n", x);
//         }
//         while(threadIdx.x < x/2) {
//             if(i == 12)
//             printf("%d\n has value %d %d and compares with %d %d", threadIdx.x, labellist[threadIdx.x].key, labellist[threadIdx.x].val, labellist[threadIdx.x+(x/2)].key, labellist[threadIdx.x+(x/2)].val);
//             labellist[threadIdx.x] = (labellist[threadIdx.x].val > labellist[threadIdx.x+(x/2)].val) ? labellist[threadIdx.x] : 
//                 labellist[threadIdx.x+(x/2)];
//             x = x/2;
//             __syncthreads();
//         }
//     }
// }
//update listener per node memory
__global__ void SLPAListener3(int * row, int *col, int *val, int * memnnz, triples *mem, triples *labellist, int n, int T) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = tid; i < n; i+= N*B) {
        int flag = 0;
        for(int j = 0; j < n; j++) {
            if(mem[n*i+j].key == labellist[row[i]].key) {
                mem[n*i+j].val++;
                flag = 1;
            }
        }
        if(flag == 0 && memnnz[i] < n) {
            mem[n*i+memnnz[i]] = labellist[row[i]];
            mem[n*i+memnnz[i]].val = 1;
            memnnz[i]++;
        }
        else if(flag == 0 && memnnz[i] >=n) {
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

__global__ void SLPAPostProcess(triples *mem, int n, int T, float r) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = tid; i < n*n; i+=N*B) { //d+1
        //printf("%d %f\n", mem[i].val, T*r);
        if(mem[i].val < (int)(T*r)) {
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

void convertToCSR(double* matrix, int *&row, int *&col, int *&val, int n) {
    row[0] = 0;
    int r = 1;
    int c = 0;
    for(int i = 0; i < n; i++) {
        row[r] = row[r-1];
        for(int j = 0; j < n; j++) {
            if(matrix[i*n+j] != 0) {
                row[r]++;
                col[c] = j;
                val[c] = matrix[i*n+j];
                c++;
            }
        }
        r++;
    }
}


int main(int argc, char *argv[]) {

    //Example graph taken from paper https://webdocs.cs.ualberta.ca/~zaiane/postscript/cason09.pdf
     int mod = 10;
    int T = 50;
    float r = 0.1;
    int n = 0;
    int cols = 0;
    int num_lines = 0;
    int * row;
    int *col, *val, *row_id, *memnnz;
    triples *mem, *labellist;
    std::cout << "malloc call\n";
    std::string arg1(argv[1]);
    std::ifstream file(arg1);

    // Ignore comments headers
    while (file.peek() == '%') file.ignore(2048, '\n');

    // Read number of rows and columns
    file >> n >> cols >> num_lines;
    std::cout << "# rows is " << n << " # cols is " << cols << " # lines is " << num_lines << "\n";

    gpuErrchk(cudaMallocManaged(&row, n * sizeof(int)));
    cudaMallocManaged(&col, num_lines*2 * sizeof(int));
    cudaMallocManaged(&val, num_lines*2 * sizeof(int));
    cudaMallocManaged(&memnnz, n * sizeof(int));
    cudaMallocManaged(&row_id, n * sizeof(int));
    cudaMallocManaged(&mem, n*n*sizeof(triples));
    cudaMallocManaged(&labellist, n*sizeof(triples));
    

    // Create 2D array and fill with zeros
    double* matrix;              
    matrix = new double[n * cols];      
    std::fill(matrix, matrix + n *cols, 0.);

    // fill the matrix with data
    for (int l = 0; l < num_lines; l++)
    {
        int row, col;
        file >> row >> col; 
        matrix[(row -1) + (col -1) * n] = 1;
        matrix[(row -1) *n + (col -1)] = 1;
    }

    file.close();

    convertToCSR(matrix, row, col, val, n);
    // for(int i = 0; i < n; i++) {
    //     std::cout << row[i] << " ";
    // }
    // std::cout << "\n";
    //exit(0);
    // row[0] = 0;
    // row[1] = 3;
    // row[2] = 7;
    // row[3] = 12;
    // row[4] = 17;
    // row[5] = 21;
    // row[6] = 26;
    // row[7] = 31;
    // row[8] = 34;
    // row[9] = 38;
    // row[10] = 42;
    // row[11] = 46;
    // row[12] = 49;
    // row[13] = 56;

    // col[0] = 1;
    // col[1] = 2;
    // col[2] = 3;
    // col[3] = 0;
    // col[4] = 2;
    // col[5] = 3;
    // col[6] = 12;
    // col[7] = 0;
    // col[8] = 1;
    // col[9] = 3;
    // col[10] = 6;
    // col[11] = 12;
    // col[12] = 0;
    // col[13] = 1;
    // col[14] = 2;
    // col[15] = 8;
    // col[16] = 12;
    // col[17] = 5;
    // col[18] = 6;
    // col[19] = 7;
    // col[20] = 12;
    // col[21] = 4;
    // col[22] = 6;
    // col[23] = 7;
    // col[24] = 9;
    // col[25] = 12;
    // col[26] = 2;
    // col[27] = 4;
    // col[28] = 5;
    // col[29] = 7;
    // col[30] = 12;
    // col[31] = 4;
    // col[32] = 5;
    // col[33] = 6;
    // col[34] = 3;
    // col[35] = 9;
    // col[36] = 10;
    // col[37] = 11;
    // col[38] = 5;
    // col[39] = 8;
    // col[40] = 10;
    // col[41] = 11;
    // col[42] = 8;
    // col[43] = 9;
    // col[44] = 11;
    // col[45] = 12;
    // col[46] = 8;
    // col[47] = 9;
    // col[48] = 10;
    // col[49] = 1;
    // col[50] = 2;
    // col[51] = 3;
    // col[52] = 4;
    // col[53] = 5;
    // col[54] = 6;
    // col[55] = 10;

    // val[0] = 1;    
    // val[1] = 1;    
    // val[2] = 1;    
    // val[3] = 1;    
    // val[4] = 1;    
    // val[5] = 1;    
    // val[6] = 1;
    // val[7] = 1;    
    // val[8] = 1;    
    // val[9] = 1;    
    // val[10] = 1;
    // val[11] = 1;
    // val[12] = 1;
    // val[13] = 1;
    // val[14] = 1;
    // val[15] = 1;
    // val[16] = 1;
    // val[17] = 1;
    // val[18] = 1;
    // val[19] = 1;
    // val[20] = 1;
    // val[21] = 1;
    // val[22] = 1;
    // val[23] = 1;
    // val[24] = 1;
    // val[25] = 1;
    // val[26] = 1;
    // val[27] = 1;
    // val[28] = 1;
    // val[29] = 1;
    // val[30] = 1;
    // val[31] = 1;
    // val[32] = 1;
    // val[33] = 1;
    // val[34] = 1;
    // val[35] = 1;
    // val[36] = 1;
    // val[37] = 1;
    // val[38] = 1;
    // val[39] = 1;
    // val[40] = 1;
    // val[41] = 1;
    // val[42] = 1;
    // val[43] = 1;
    // val[44] = 1;
    // val[45] = 1;
    // val[46] = 1;
    // val[47] = 1;
    // val[48] = 1;
    // val[49] = 1;
    // val[50] = 1;
    // val[51] = 1;
    // val[52] = 1;
    // val[53] = 1;
    // val[54] = 1;
    // val[55] = 1;

    // row_id[0] = 0;    
    // row_id[1] = 0;    
    // row_id[2] = 0;    
    // row_id[3] = 1;    
    // row_id[4] = 1;    
    // row_id[5] = 1;    
    // row_id[6] = 1;
    // row_id[7] = 2;    
    // row_id[8] = 2;    
    // row_id[9] = 2;    
    // row_id[10] = 2;
    // row_id[11] = 2;
    // row_id[12] = 3;
    // row_id[13] = 3;
    // row_id[14] = 3;
    // row_id[15] = 3;
    // row_id[16] = 3;
    // row_id[17] = 4;
    // row_id[18] = 4;
    // row_id[19] = 4;
    // row_id[20] = 4;
    // row_id[21] = 5;
    // row_id[22] = 5;
    // row_id[23] = 5;
    // row_id[24] = 5;
    // row_id[25] = 5;
    // row_id[26] = 6;
    // row_id[27] = 6;
    // row_id[28] = 6;
    // row_id[29] = 6;
    // row_id[30] = 6;
    // row_id[31] = 7;
    // row_id[32] = 7;
    // row_id[33] = 7;
    // row_id[34] = 8;
    // row_id[35] = 8;
    // row_id[36] = 8;
    // row_id[37] = 8;
    // row_id[38] = 9;
    // row_id[39] = 9;
    // row_id[40] = 9;
    // row_id[41] = 9;
    // row_id[42] = 10;
    // row_id[43] = 10;
    // row_id[44] = 10;
    // row_id[45] = 10;
    // row_id[46] = 11;
    // row_id[47] = 11;
    // row_id[48] = 11;
    // row_id[49] = 12;
    // row_id[50] = 12;
    // row_id[51] = 12;
    // row_id[52] = 12;
    // row_id[53] = 12;
    // row_id[54] = 12;
    // row_id[55] = 12;

    for(int i = 0; i < cols; i++) {
        labellist[i].key = -1;
        labellist[i].val = 0;
    }

    for(int i = 0; i < n; i++) {
        memnnz[i] = 1;
        for(int j = 0; j < n; j++) {
            mem[i*n + j].key = i;
            mem[i*n + j].val = 1;
        }
    }

    curandState *dev_random;
    cudaMallocManaged(&dev_random, N*B*sizeof(curandState));

    for(int i = 1; i < T; i++) {
        //  if(i%mod == 0) {
        // std::cout << "memnnz values per vertex:\n";
        // for(int j = 0; j < n; j++) {
        //     std::cout << j << ":" << memnnz[j] << "  ";
        // }
        // std::cout << "\n";
        //  }
        // //speaker rule
        // if(DEBUGS) {
        // std::cout << time(NULL) << "\n";
        // }
        SLPASpeaker<<<B,N>>>(row,col,val, memnnz, row_id, mem, labellist, n, T, dev_random, time(NULL)+i);
        cudaDeviceSynchronize();
        if(DEBUGS) {
            if(i%mod == 0) {
                std::cout << "finished iteration after speaker    " << i << "\n";
                for(int k = 12; k < n; k++) {
                    std::cout << "Row " << k << " labellist\n";
                    for(int j = row[k]; j < row[k+1]; j++) {
                        std::cout << labellist[j].key << " " << labellist[j].val << "\t\t\t";
                    }
                    std::cout << "\n";
                }
            }
        }
        //label frequency
        SLPAListener1<<<B,N>>>(row,col,val, labellist, row_id, n);
        cudaDeviceSynchronize();
        if(DEBUG1) {
            if(i%mod == 0) {
                std::cout << "finished iteration after listener 1    " << i << "\n";
                for(int k = 12; k < n; k++) {
                    std::cout << "Row " << k << " labellist\n";
                    for(int j = row[k]; j < row[k+1]; j++) {
                        std::cout << labellist[j].key << " " << labellist[j].val << "\t\t\t";
                    }
                    std::cout << "\n";
                }
            }
        }
        //max frequency
        SLPAListener2<<<B,N>>>(row,col,val, labellist, n, row_id);
        cudaDeviceSynchronize();
        if(DEBUG2) {
         if(i%mod == 0) {
            std::cout << "finished iteration after listener 2    " << i << "\n";
            for(int k = 12; k < n; k++) {
                std::cout << "Row " << k << " labellist\n";
                for(int j = row[k]; j < row[k+1]; j++) {
                    std::cout << labellist[j].key << " " << labellist[j].val << "\t\t\t";
                }
                std::cout << "\n";
            }
        }
        }
        int * labellistnnz;
        cudaMallocManaged(&labellistnnz, n*sizeof(int));
        for(int k = 0; k < n; k++) {
            labellistnnz[k] = 1;
        }
        //check for tiebreak
        SLPAListener2_1<<<B,N>>>(row, labellist, n, row_id, labellistnnz);
        cudaDeviceSynchronize();
        if(DEBUG2) {
         if(i%mod == 0) {
            std::cout << "finished iteration after listener 2_1   " << i << "\n";
            for(int k = 12; k < n; k++) {
                std::cout << "Row " << k << " labellist\n";
                for(int j = row[k]; j < row[k+1]; j++) {
                    std::cout << labellist[j].key << " " << labellist[j].val << "\t\t\t";
                }
                std::cout << "\n";
            }
        }
        }

        //perform tiebreak
        SLPAListener2_2<<<B,N>>>(row, labellist, n, row_id, labellistnnz, dev_random, time(NULL)+i);
        cudaDeviceSynchronize();
        if(DEBUG2) {
         if(i%mod == 0) {
            std::cout << "finished iteration after listener 2_2    " << i << "\n";
            for(int k = 12; k < n; k++) {
                std::cout << "Row " << k << " labellist\n";
                for(int j = row[k]; j < row[k+1]; j++) {
                    std::cout << labellist[j].key << " " << labellist[j].val << "\t\t\t";
                }
                std::cout << "\n";
            }
        }
        }

        //memory update
        SLPAListener3<<<B,N>>>(row,col,val, memnnz, mem, labellist, n, T);
        cudaDeviceSynchronize();
        if(DEBUG3) {
            //std::cout << "Finished iteration " << i << "\n";
            if(i%mod == 0) {
                std::cout << "finished iteration " << i << "\n";
                for(int k = 0; k < n; k++) {
                    std::cout << "vertex " << k << "\t";
                    for(int j = 0; j < memnnz[k]; j++) {
                        std::cout << mem[n*k+j].key << " " << mem[n*k+j].val << "\t";
                    }
                    std::cout << "\n";
                }
            }
        }
    }
    
    SLPAPostProcess<<<B,N>>>(mem, n, T, r);
    cudaDeviceSynchronize();
    //SLPA<<<B,N>>>(row,col,val,mem, n, T,r, dev_random);
    for(int i = 0; i < n; i++) {
        int count = 0;
        std::cout << "Vertex " << i << " is in communities ";
        for(int j = 0; j < memnnz[i]; j++) {
            if(mem[i*n+j].key != -1) {
                std::cout << mem[i*n+j].key << " ";
                count++;
            }
        }
        if(count > 1)
            std::cout << "overlap\n";
        else
            std::cout << " only community\n";
    }
    std::cout << "\n";
    return 0;
}

// int main(int argc, char *argv[]) {

//     //Example graph taken from paper https://webdocs.cs.ualberta.ca/~zaiane/postscript/cason09.pdf

//     int mod = 10;
//     int T = 50;
//     float r = 0.1;
//     int n = 13;
//     int cols = 56;
//     int num_lines = ((float)56/(float)2);
//     int * row;
//     int *col, *val, *row_id, *memnnz;
//     triples *mem, *labellist;
//     std::cout << "malloc call\n";

//     gpuErrchk(cudaMallocManaged(&row, n * sizeof(int)));
//     cudaMallocManaged(&col, cols * sizeof(int));
//     cudaMallocManaged(&val, cols * sizeof(int));
//     cudaMallocManaged(&memnnz, n * sizeof(int));
//     cudaMallocManaged(&row_id, n * sizeof(int));
//     cudaMallocManaged(&mem, n*n*sizeof(triples));
//     cudaMallocManaged(&labellist, cols*sizeof(triples));
    

//     row[0] = 0;
//     row[1] = 3;
//     row[2] = 7;
//     row[3] = 12;
//     row[4] = 17;
//     row[5] = 21;
//     row[6] = 26;
//     row[7] = 31;
//     row[8] = 34;
//     row[9] = 38;
//     row[10] = 42;
//     row[11] = 46;
//     row[12] = 49;
//     row[13] = 56;

//     col[0] = 1;
//     col[1] = 2;
//     col[2] = 3;
//     col[3] = 0;
//     col[4] = 2;
//     col[5] = 3;
//     col[6] = 12;
//     col[7] = 0;
//     col[8] = 1;
//     col[9] = 3;
//     col[10] = 6;
//     col[11] = 12;
//     col[12] = 0;
//     col[13] = 1;
//     col[14] = 2;
//     col[15] = 8;
//     col[16] = 12;
//     col[17] = 5;
//     col[18] = 6;
//     col[19] = 7;
//     col[20] = 12;
//     col[21] = 4;
//     col[22] = 6;
//     col[23] = 7;
//     col[24] = 9;
//     col[25] = 12;
//     col[26] = 2;
//     col[27] = 4;
//     col[28] = 5;
//     col[29] = 7;
//     col[30] = 12;
//     col[31] = 4;
//     col[32] = 5;
//     col[33] = 6;
//     col[34] = 3;
//     col[35] = 9;
//     col[36] = 10;
//     col[37] = 11;
//     col[38] = 5;
//     col[39] = 8;
//     col[40] = 10;
//     col[41] = 11;
//     col[42] = 8;
//     col[43] = 9;
//     col[44] = 11;
//     col[45] = 12;
//     col[46] = 8;
//     col[47] = 9;
//     col[48] = 10;
//     col[49] = 1;
//     col[50] = 2;
//     col[51] = 3;
//     col[52] = 4;
//     col[53] = 5;
//     col[54] = 6;
//     col[55] = 10;

//     val[0] = 1;    
//     val[1] = 1;    
//     val[2] = 1;    
//     val[3] = 1;    
//     val[4] = 1;    
//     val[5] = 1;    
//     val[6] = 1;
//     val[7] = 1;    
//     val[8] = 1;    
//     val[9] = 1;    
//     val[10] = 1;
//     val[11] = 1;
//     val[12] = 1;
//     val[13] = 1;
//     val[14] = 1;
//     val[15] = 1;
//     val[16] = 1;
//     val[17] = 1;
//     val[18] = 1;
//     val[19] = 1;
//     val[20] = 1;
//     val[21] = 1;
//     val[22] = 1;
//     val[23] = 1;
//     val[24] = 1;
//     val[25] = 1;
//     val[26] = 1;
//     val[27] = 1;
//     val[28] = 1;
//     val[29] = 1;
//     val[30] = 1;
//     val[31] = 1;
//     val[32] = 1;
//     val[33] = 1;
//     val[34] = 1;
//     val[35] = 1;
//     val[36] = 1;
//     val[37] = 1;
//     val[38] = 1;
//     val[39] = 1;
//     val[40] = 1;
//     val[41] = 1;
//     val[42] = 1;
//     val[43] = 1;
//     val[44] = 1;
//     val[45] = 1;
//     val[46] = 1;
//     val[47] = 1;
//     val[48] = 1;
//     val[49] = 1;
//     val[50] = 1;
//     val[51] = 1;
//     val[52] = 1;
//     val[53] = 1;
//     val[54] = 1;
//     val[55] = 1;

//     row_id[0] = 0;    
//     row_id[1] = 0;    
//     row_id[2] = 0;    
//     row_id[3] = 1;    
//     row_id[4] = 1;    
//     row_id[5] = 1;    
//     row_id[6] = 1;
//     row_id[7] = 2;    
//     row_id[8] = 2;    
//     row_id[9] = 2;    
//     row_id[10] = 2;
//     row_id[11] = 2;
//     row_id[12] = 3;
//     row_id[13] = 3;
//     row_id[14] = 3;
//     row_id[15] = 3;
//     row_id[16] = 3;
//     row_id[17] = 4;
//     row_id[18] = 4;
//     row_id[19] = 4;
//     row_id[20] = 4;
//     row_id[21] = 5;
//     row_id[22] = 5;
//     row_id[23] = 5;
//     row_id[24] = 5;
//     row_id[25] = 5;
//     row_id[26] = 6;
//     row_id[27] = 6;
//     row_id[28] = 6;
//     row_id[29] = 6;
//     row_id[30] = 6;
//     row_id[31] = 7;
//     row_id[32] = 7;
//     row_id[33] = 7;
//     row_id[34] = 8;
//     row_id[35] = 8;
//     row_id[36] = 8;
//     row_id[37] = 8;
//     row_id[38] = 9;
//     row_id[39] = 9;
//     row_id[40] = 9;
//     row_id[41] = 9;
//     row_id[42] = 10;
//     row_id[43] = 10;
//     row_id[44] = 10;
//     row_id[45] = 10;
//     row_id[46] = 11;
//     row_id[47] = 11;
//     row_id[48] = 11;
//     row_id[49] = 12;
//     row_id[50] = 12;
//     row_id[51] = 12;
//     row_id[52] = 12;
//     row_id[53] = 12;
//     row_id[54] = 12;
//     row_id[55] = 12;

//     for(int i = 0; i < cols; i++) {
//         labellist[i].key = -1;
//         labellist[i].val = 0;
//     }

//     for(int i = 0; i < n; i++) {
//         memnnz[i] = 1;
//         for(int j = 0; j < n; j++) {
//             mem[i*n + j].key = i;
//             mem[i*n + j].val = 1;
//         }
//     }

//     curandState *dev_random;
//     cudaMallocManaged(&dev_random, N*B*sizeof(curandState));

//     for(int i = 1; i < T; i++) {
//         //  if(i%mod == 0) {
//         // std::cout << "memnnz values per vertex:\n";
//         // for(int j = 0; j < n; j++) {
//         //     std::cout << j << ":" << memnnz[j] << "  ";
//         // }
//         // std::cout << "\n";
//         //  }
//         // //speaker rule
//         // if(DEBUGS) {
//         // std::cout << time(NULL) << "\n";
//         // }
//         SLPASpeaker<<<B,N>>>(row,col,val, memnnz, row_id, mem, labellist, n, T, dev_random, time(NULL)+i);
//         cudaDeviceSynchronize();
//         if(DEBUGS) {
//             if(i%mod == 0) {
//                 std::cout << "finished iteration after speaker    " << i << "\n";
//                 for(int k = 12; k < n; k++) {
//                     std::cout << "Row " << k << " labellist\n";
//                     for(int j = row[k]; j < row[k+1]; j++) {
//                         std::cout << labellist[j].key << " " << labellist[j].val << "\t\t\t";
//                     }
//                     std::cout << "\n";
//                 }
//             }
//         }
//         //label frequency
//         SLPAListener1<<<B,N>>>(row,col,val, labellist, row_id, n);
//         cudaDeviceSynchronize();
//         if(DEBUG1) {
//             if(i%mod == 0) {
//                 std::cout << "finished iteration after listener 1    " << i << "\n";
//                 for(int k = 12; k < n; k++) {
//                     std::cout << "Row " << k << " labellist\n";
//                     for(int j = row[k]; j < row[k+1]; j++) {
//                         std::cout << labellist[j].key << " " << labellist[j].val << "\t\t\t";
//                     }
//                     std::cout << "\n";
//                 }
//             }
//         }
//         //max frequency
//         SLPAListener2<<<B,N>>>(row,col,val, labellist, n, row_id);
//         cudaDeviceSynchronize();
//         if(DEBUG2) {
//          if(i%mod == 0) {
//             std::cout << "finished iteration after listener 2    " << i << "\n";
//             for(int k = 12; k < n; k++) {
//                 std::cout << "Row " << k << " labellist\n";
//                 for(int j = row[k]; j < row[k+1]; j++) {
//                     std::cout << labellist[j].key << " " << labellist[j].val << "\t\t\t";
//                 }
//                 std::cout << "\n";
//             }
//         }
//         }
//         int * labellistnnz;
//         cudaMallocManaged(&labellistnnz, n*sizeof(int));
//         for(int k = 0; k < n; k++) {
//             labellistnnz[k] = 1;
//         }
//         //check for tiebreak
//         SLPAListener2_1<<<B,N>>>(row, labellist, n, row_id, labellistnnz);
//         cudaDeviceSynchronize();
//         if(DEBUG2) {
//          if(i%mod == 0) {
//             std::cout << "finished iteration after listener 2_1    " << i << "\n";
//             for(int k = 12; k < n; k++) {
//                 std::cout << "Row " << k << " labellist\n";
//                 for(int j = row[k]; j < row[k+1]; j++) {
//                     std::cout << labellist[j].key << " " << labellist[j].val << "\t\t\t";
//                 }
//                 std::cout << "\n";
//             }
//         }
//         }

//         //perform tiebreak
//         SLPAListener2_2<<<B,N>>>(row, labellist, n, row_id, labellistnnz, dev_random, time(NULL)+i);
//         cudaDeviceSynchronize();
//         if(DEBUG2) {
//          if(i%mod == 0) {
//             std::cout << "finished iteration after listener 2_2    " << i << "\n";
//             for(int k = 12; k < n; k++) {
//                 std::cout << "Row " << k << " labellist\n";
//                 for(int j = row[k]; j < row[k+1]; j++) {
//                     std::cout << labellist[j].key << " " << labellist[j].val << "\t\t\t";
//                 }
//                 std::cout << "\n";
//             }
//         }
//         }

//         //memory update
//         SLPAListener3<<<B,N>>>(row,col,val, memnnz, mem, labellist, n, T);
//         cudaDeviceSynchronize();
//         if(DEBUG3) {
//             //std::cout << "Finished iteration " << i << "\n";
//             if(i%mod == 0) {
//                 std::cout << "finished iteration " << i << "\n";
//                 for(int k = 0; k < n; k++) {
//                     std::cout << "vertex " << k << "\t";
//                     for(int j = 0; j < memnnz[k]; j++) {
//                         std::cout << mem[n*k+j].key << " " << mem[n*k+j].val << "\t";
//                     }
//                     std::cout << "\n";
//                 }
//             }
//         }
//     }
    
//     SLPAPostProcess<<<B,N>>>(mem, n, T, r);
//     cudaDeviceSynchronize();
//     //SLPA<<<B,N>>>(row,col,val,mem, n, T,r, dev_random);
//     for(int i = 0; i < n; i++) {
//         int count = 0;
//         std::cout << "Vertex " << i << " is in communities ";
//         for(int j = 0; j < memnnz[i]; j++) {
//             if(mem[i*n+j].key != -1) {
//                 std::cout << mem[i*n+j].key << " ";
//                 count++;
//             }
//         }
//         if(count > 1)
//             std::cout << "overlap\n";
//         else
//             std::cout << " only community\n";
//     }
//     std::cout << "\n";
//     return 0;
// }