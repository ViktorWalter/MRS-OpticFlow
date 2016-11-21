#define arraySize 50
#define MinValThreshold (scanRadius*scanRadius*0.2)

__kernel void OptFlow_C1_D0(	__constant unsigned char* input_1,
                                __constant unsigned char* input_2,
                                int imgSrcWidth,
                                int imgDstWidth,
                                __global signed char* output_X,
                                __global signed char* output_Y,
                                int blockSize,
                                int blockStep,
                                int scanRadius)
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
                                 input_1[((blockX*(blockSize+blockStep)) + scanRadius + i)+
                                         ((blockY*(blockSize+blockStep)) + scanRadius + j)*imgSrcWidth]
                                 -
                                 input_2[((blockX*(blockSize+blockStep)) + i + threadX)+
                                         ((blockY*(blockSize+blockStep)) + j + threadY)*imgSrcWidth]
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
           for (int i=1;i<ScanDiameter;i++)
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
           for (int i=1;i<ScanDiameter;i++)
              {
                 if (minvalFin > minval[i])
                 {
                    minvalFin = minval[i];
                    minY = i-scanRadius;
                 }
              }
           output_Y[blockY*imgDstWidth+blockX] = minY;
           output_X[blockY*imgDstWidth+blockX] = minX[minY+scanRadius];

           if ((abssum[scanRadius][scanRadius] - minvalFin) <= MinValThreshold)  //if the difference is small, then it is considered to be noise in a uniformly colored area
           {
              output_Y[blockY*imgDstWidth+blockX]= 0;
              output_X[blockY*imgDstWidth+blockX] = 0;
           }

        }

}

__kernel void Histogram_C1_D0(__constant signed char* input,
                                  int scanRadius,
                                  int ScanDiameter,
                                  signed char value)
{

        int blockX = get_group_id(0);
        int blockY = get_group_id(1);
        int threadX = get_local_id(0);
        int threadY = get_local_id(1);

        __local int Histogram[arraySize];


    int index = (threadY*get_local_size(0)+threadX);
    if (index < ScanDiameter)
        Histogram[index] = 0;

    __syncthreads();

    atomic_add(&(Histogram[input[index]+scanRadius]),1);

    __syncthreads();


    if ( (threadY == 0) && (threadX == 0))
    {
        int MaxIndex = 0;
        char  MaxVal = 0;

        for (int i=0;i<ScanDiameter;i++)
        {
            if (MaxVal < Histogram[i])
            {
                MaxVal = Histogram[i];
                MaxIndex = i;
            }
        }
        value = MaxIndex - scanRadius;


    }

}

