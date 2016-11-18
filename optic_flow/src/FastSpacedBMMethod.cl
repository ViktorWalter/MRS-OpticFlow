#define arraySize 50
#define MinValThreshold (scanRadius*scanRadius*0.2)

__kernel void OptFlow_C1_D0(	__constant unsigned char* input_1,
                                __constant unsigned char* input_2,
                                __global unsigned char* output_X,
                                __global unsigned char* output_Y,
                                int blockSize,
                                int blockStep,
                                int scanRadius,
                                int width,
                                int height)
{
        int blockX = get_group_id(0);
        int blockY = get_group_id(1);
        int threadX = get_local_id(0);
        int threadY = get_local_id(1);


        int ScanDiameter = scanRadius*2+1;
        __local int abssum[arraySize][arraySize];

        abssum[threadY][threadX] = 0;

            for (int i=0;i<blockSize;i++)
            {
                for (int j=0;j<blockSize;j++)
                {
                    atomic_add(&(abssum[threadY][threadX]),
                            abs(
                                atomic_sub(
                                 input_1(((blockX*(blockSize+blockStep)) + scanRadius + i),
                                         ((blockY*(blockSize+blockStep)) + scanRadius + j))
                                 ,
                                 input_2(((blockX*(blockSize+blockStep)) + i + threadY),
                                         ((blockY*(blockSize+blockStep)) + j + threadX))
                                 )
                            );
                }

            }

        barrier(CLK_LOCAL_MEM_FENCE);

        __local int minval[arraySize];
        __local signed char minX[arraySize];
        signed char minY;

        if (threadY == 0)
        {
           minval[threadX] = abssum[threadX][0];
           minX[threadX] = -scanRadius;
           for (int i=1;i<scanDiameter;i++)
           {
              if (minval[threadX] > abssum[threadX][i])
              {
                 minval[threadX] = abssum[threadX][i];
                 minX[threadX] = i-scanRadius;
              }
           }
        }

        barrier(CLK_LOCAL_MEM_FENCE);


        if ( (threadY == 0) && (threadX == 0))
        {
           int minvalFin = minval[0];
           minY = -scanRadius;
           for (int i=1;i<scanDiameter;i++)
              {
                 if (minvalFin > minval[i])
                 {
                    minvalFin = minval[i];
                    minY = i-scanRadius;
                 }
              }
           output_Y(blockY,blockX) = minY;
           output_X(blockY,blockX) = minX[minY+scanRadius];

           if ((abssum[scanRadius][scanRadius] - minvalFin) <= MinValThreshold)  //if the difference is small, then it is considered to be noise in a uniformly colored area
           {
              output_Y(blockY,blockX) = 0;
              output_X(blockY,blockX) = 0;
           }

        }

}

__kernel void Histogram(__constant signed char input,
                                  int scanRadius,
                                  int scanDiameter,
                                  signed char value)
{

        int blockX = get_group_id(0);
        int blockY = get_group_id(1);
        int threadX = get_local_id(0);
        int threadY = get_local_id(1);

        __local int Histogram[arraySize];


    int index = (threadY*blockDim.x+threadX);
    if (index < scanDiameter)
        Histogram[index] = 0;

    __syncthreads();

    atomicAdd(&(Histogram[input(threadY,threadX)+scanRadius]),1);

    __syncthreads();


    if ( (threadY == 0) && (threadX == 0))
    {
        int MaxIndex = 0;
        char  MaxVal = 0;

        for (int i=0;i<scanDiameter;i++)
        {
            if (MaxVal < Histogram[i])
            {
                MaxVal = Histogram[i];
                MaxIndex = i;
            }
        }
        *value = MaxIndex - scanRadius;


    }

}
/
