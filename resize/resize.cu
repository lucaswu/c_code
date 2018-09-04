#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <cstddef> 
#include <iostream> 
#include <memory.h> 
#include <cuda.h>

using namespace cv; 
 
#ifndef uint8_t 
#define uint8_t unsigned char 
#endif 

#define GWS(a,b)  (((a)+(b)-1)/(b)) 

#define X -1
#define Y -1



__global__ void resizePlaneData_gpu(float *pSrc,float*pDst,int srcH,int srcW,int dstH,int dstW)
{

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int idy = blockDim.y*blockIdx.y + threadIdx.y;

    float scale_x = (float)srcW/dstW;
    float scale_y = (float)srcH/dstH;

    if(idx<dstW && idy<dstH){
        float fy = (float)((idy + 0.5) * scale_y - 0.5);
        float fx = (float)((idx + 0.5) * scale_x - 0.5);
        int sy = floor(fy);
        int sx = floor(fx);

        fy -= sy;
        fx -= sx;

        sy = max(0, min(sy, srcH - 2));
        if (sx < 0) {
            fx = 0;
            sx = 0;
        }
        if (sx >= srcW - 1) {
            fx = 0;
            sx = srcW - 2;
        }

        int cbufy0, cbufy1, cbufx0, cbufx1;
        cbufy0 = ceil((1.f - fy) * 2048);
        cbufx0 = ceil((1.f - fx) * 2048);
        cbufy1 = 2048 - cbufy0;
        cbufx1 = 2048 - cbufx0;


        float* p = pDst + idy*dstW+idx;
        #pragma unroll
        for(int i=0;i<3;i++){
            float *a = pSrc + i*srcW*srcH + sy*srcW + sx;
            float *b = a + srcW;
            float *c = a + 1;
            float *d = b + 1;

            int value = a[0]*cbufx0*cbufy0 + b[0]*cbufx0*cbufy1 +c[0]*cbufx1*cbufy0 + d[0]*cbufx1*cbufy1;
            *(p+i*dstW*dstH) =  value>>22;
        }

    }
}

__global__ void resize_gpu(float *pSrc,float*pDst,int srcH,int srcW,int dstH,int dstW)
{

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int idy = blockDim.y*blockIdx.y + threadIdx.y;

    float scale_x = (float)srcW/dstW;
    float scale_y = (float)srcH/dstH;

    if(idx<dstW && idy<dstH){
        float fy = (float)((idy + 0.5) * scale_y - 0.5);
        float fx = (float)((idx + 0.5) * scale_x - 0.5);
        int sy = floor(fy);
        int sx = floor(fx);

        fy -= sy;
        fx -= sx;

        sy = max(0, min(sy, srcH - 2));
        if (sx < 0) {
            fx = 0;
            sx = 0;
        }
        if (sx >= srcW - 1) {
            fx = 0;
            sx = srcW - 2;
        }

        int cbufy0, cbufy1, cbufx0, cbufx1;
        cbufy0 = (int)((1.f - fy) * 2048+0.5f);
        cbufx0 = (int)((1.f - fx) * 2048+0.5f);
        cbufy1 = 2048 - cbufy0;
        cbufx1 = 2048 - cbufx0;


        float *a = pSrc + sy*srcW*3 + sx*3;
        float *b = a + 3;
        float *c = a + 3*srcW;
        float *d = c+ 3;

        int red   = a[0] * cbufx0*cbufy0 + b[0] * cbufx1*cbufy0 + c[0] * cbufx0*cbufy1 + d[0] * cbufx1*cbufy1;
        int green = a[1] * cbufx0*cbufy0 + b[1] * cbufx1*cbufy0 + c[1] * cbufx0*cbufy1 + d[1] * cbufx1*cbufy1;
        int blue  = a[2] * cbufx0*cbufy0 + b[2] * cbufx1*cbufy0 + c[2] * cbufx0*cbufy1 + d[2] * cbufx1*cbufy1;

        int index = idy*dstW+idx;
        pDst[index]                 = red>>22;
        pDst[index + dstW*dstH]     = green>>22;
        pDst[index + dstW*dstH*2]   = blue>>22;

    }
}

void writeFloatData2Txt( uint8_t*data,int w,int h,int channelNum,const char*filename)
{
    printf("%s at %d,write %s,(%d,%d,%d)",__FILE__,__LINE__,filename,w,h,channelNum);
    FILE *fp = NULL;
    fp = fopen(filename,"w");
    // float* pData = data;
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            fprintf(fp,"[ ");
             for(int cn = 0;cn<channelNum;cn++){
                fprintf(fp," %d ",*data++);
                // pData++;
            }
            fprintf(fp," ]");
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}


void Resize(const Mat src, Mat &dst) {
    uchar* dataDst = dst.data;
    int stepDst = dst.step;
    uchar* dataSrc = src.data;
    int stepSrc = src.step;
    int iWidthSrc = src.cols;
    int iHiehgtSrc = src.rows;

    double scale_x = (double)src.cols / dst.cols;
    double scale_y = (double)src.rows / dst.rows;



    for (int j = 0; j < dst.rows; ++j)
    {
        float fy = (float)((j + 0.5) * scale_y - 0.5);
        if (j == Y) {
            printf("scale_x=%f,scale_y=%f\n", scale_x, scale_y);
            printf("fy=%f\n", fy);
        }
            
        int sy = cvFloor(fy);
        fy -= sy;
        sy = std::min(sy, iHiehgtSrc - 2);
        sy = std::max(0, sy);
       
        short cbufy[2];
        cbufy[0] = cv::saturate_cast<short>((1.f - fy) * 2048);
        cbufy[1] = 2048 - cbufy[0];

        if(j==Y )
            printf("sy=%d,fy=%f,cbufy0=%d(%f),cbufy1=%d\n",sy,fy, cbufy[0],(1.f - fy) * 2048, cbufy[1]);

        for (int i = 0; i < dst.cols; ++i)
        {
            float fx = (float)((i + 0.5) * scale_x - 0.5);
            if (j == Y && i == X)
            {
                printf("fx=%f\n", fx);
            }
            int sx = cvFloor(fx);
            fx -= sx;

            if (sx < 0) {
                fx = 0, sx = 0;
            }
            if (sx >= iWidthSrc - 1) {
                fx = 0, sx = iWidthSrc - 2;
            }

            short cbufx[2];
            cbufx[0] = cv::saturate_cast<short>((1.f - fx) * 2048);
            cbufx[1] = 2048 - cbufx[0];

            if (j == Y && i == X)
                printf("sx=%d,fx=%f,cbufx0=%d(%f),cbufx1=%d\n", sx, fx, cbufx[0],(1.f - fx) * 2048, cbufx[1]);

            for (int k = 0; k < dst.channels(); ++k)
            {
                *(dataDst + j*stepDst + 3 * i + k) = (*(dataSrc + sy*stepSrc + 3 * sx + k) * cbufx[0] * cbufy[0] +
                    *(dataSrc + (sy + 1)*stepSrc + 3 * sx + k) * cbufx[0] * cbufy[1] +
                    *(dataSrc + sy*stepSrc + 3 * (sx + 1) + k) * cbufx[1] * cbufy[0] +
                    *(dataSrc + (sy + 1)*stepSrc + 3 * (sx + 1) + k) * cbufx[1] * cbufy[1]) >> 22;

                if (j == Y && i == X)
                {
                   printf("chanel=%d:(%d(%d)(%d),%d(%d)(%d),%d(%d)(%d),%d(%d)(%d))=%d\n",k, 
                    *(dataSrc + sy*stepSrc + 3 * sx + k),sy*stepSrc + 3 * sx + k,*(dataSrc + sy*stepSrc + 3 * sx + k) * cbufx[0] * cbufy[0],
                     *(dataSrc + (sy + 1)*stepSrc + 3 * sx + k),(sy + 1)*stepSrc + 3 * sx + k,*(dataSrc + (sy + 1)*stepSrc + 3 * sx + k) * cbufx[0] * cbufy[1],
                     *(dataSrc + sy*stepSrc + 3 * (sx + 1) + k),sy*stepSrc + 3 * (sx + 1) + k,*(dataSrc + sy*stepSrc + 3 * (sx + 1) + k) * cbufx[1] * cbufy[0],
                    *(dataSrc + (sy + 1)*stepSrc + 3 * (sx + 1) + k),(sy + 1)*stepSrc + 3 * (sx + 1) + k, *(dataSrc + (sy + 1)*stepSrc + 3 * (sx + 1) + k) * cbufx[1] * cbufy[1],
                        (*(dataDst + j*stepDst + 3 * i + k)));

                }
            }
        }
    }

    
}


void gpuResize(float*pSrc,float*pDst,int srcW,int srcH,int dstW,int dstH,int dataType)
{
    dim3 threadPerBlock(16, 8);
    dim3 blocksPerGrid(GWS(dstW, threadPerBlock.x), GWS(dstH, threadPerBlock.y));
    if(dataType == 0)
        resizePlaneData_gpu<<<blocksPerGrid,threadPerBlock>>>(pSrc,pDst,srcH,srcW,dstH,dstW);  
    else
        resize_gpu<<<blocksPerGrid,threadPerBlock>>>(pSrc,pDst,srcH,srcW,dstH,dstW);  
}



void writeFloatData2TxtFloat( float*data,int w,int h,int channelNum,const char*filename)
{
    printf("%s at %d,write %s,(%d,%d,%d)",__FILE__,__LINE__,filename,w,h,channelNum);
    FILE *fp = NULL;
    fp = fopen(filename,"w");
    // float* pData = data;
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            fprintf(fp,"[ ");
             for(int cn = 0;cn<channelNum;cn++){
                fprintf(fp," %f ",data[cn*w*h+i*w+j]);
                // pData++;
            }
            fprintf(fp," ]");
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

int main(int argc, char**argv)
{
    Mat img = imread(argv[1]);

    int newH = 234;
    int newW = 416;
    Mat out(newH,newW,CV_8UC3);

    Resize(img,out);
    imwrite(argv[2],out);
    

    float* pCPU ,*pGPU;
    float* pCPUResize,*pGPUResize;

    int height = img.rows; 
    int width = img.cols; 
    int channel = img.channels();

    size_t size = height*width*channel*sizeof(float);
    pCPU = (float*)malloc(size);

    cudaMalloc((void**)&pGPU,size);

    uint8_t *pData =  img.data;

    int patchSize = width*height;
    for(int i = 0 ;i <height;i++){
        for(int j=0;j<width;j++){
            pCPU[i*width+j]                = pData[i*width*channel+j*channel];
            pCPU[i*width+j+patchSize]      = pData[i*width*channel+j*channel+1];
            pCPU[i*width+j+patchSize*2]    = pData[i*width*channel+j*channel+2];
        }
    }

    // writeFloatData2Txt(img.data,width,height,3,"cpuInput.dat");
    // writeFloatData2TxtFloat(pCPU,width,height,channel,"gpuInput.dat");

    cudaMemcpy(pGPU,pCPU,size,cudaMemcpyHostToDevice);

    size = newW*newH*3*sizeof(float);
    pCPUResize = (float*)malloc(size);
    cudaMalloc((void**)&pGPUResize,size);


    gpuResize(pGPU,pGPUResize,width,height,newW,newH,0);
    cudaMemcpy(pCPUResize,pGPUResize,size,cudaMemcpyDeviceToHost);


    pData = out.data;
    for(int i=0;i<newH;i++){
        for(int j=0;j<newW;j++){
            for(int c= 0;c<3;c++){
                int gpuIndex = c*newW*newH+i*newW+j;
                int cpuIndex = i*newW*3 + j*3 +c;
                if(abs(pCPUResize[gpuIndex] - pData[cpuIndex])>2){
                   printf("111:%s:%d,error at (%d,%d,%d)=%f,%d\n",__FILE__,__LINE__,i,j,c,pCPUResize[gpuIndex],pData[cpuIndex]);
                   return -1;
                }
            }
        }
    }


    
    pData =  img.data;
    for(int i=0;i<width*height*channel;i++){
        pCPU[i] = pData[i];
    }
    size = height*width*channel*sizeof(float);

    cudaMemcpy(pGPU,pCPU,size,cudaMemcpyHostToDevice);
    gpuResize(pGPU,pGPUResize,width,height,newW,newH,1);
    cudaMemcpy(pCPUResize,pGPUResize,size,cudaMemcpyDeviceToHost);
    pData = out.data;
    for(int i=0;i<newH;i++){
        for(int j=0;j<newW;j++){
            for(int c= 0;c<3;c++){
                int gpuIndex = c*newW*newH+i*newW+j;
                int cpuIndex = i*newW*3 + j*3 +c;
                if(abs(pCPUResize[gpuIndex] - pData[cpuIndex])>2){
                    printf("%s:%d,error at (%d,%d,%d)=%f,%d\n",__FILE__,__LINE__,i,j,c,pCPUResize[gpuIndex],pData[cpuIndex]);
                    return -1;
                }
            }
        }
    }

    free(pCPU);
    free(pCPUResize);
    cudaFree(pGPU);
    cudaFree(pGPUResize);
}
