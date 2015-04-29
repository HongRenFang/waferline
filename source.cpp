#include <opencv2/opencv.hpp> 
#include <opencv2/imgproc/imgproc_c.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h> 
#include <algorithm>
#include <omp.h>
using namespace std;
using namespace cv;
void prob(Mat& img,double p[256]);
void eq(  Mat& imgray, Mat& imgrayeq);
void otsu_method( Mat& imgray, Mat imgb,int bound);
void bubblesort(int *arr,int *index, int n);

int main()
{
	Mat img=imread("5.bmp",0); //load image
	Mat imgray,imgblur,imgblur2,imgb,result;//declare image array variable
	Mat imgb2;
	Mat histogram,equalization,equalization2; //histogram
	int r,g,b,gray;
	int x[256]={0};//histogram value
	int x_dif[257]={0},x_dif2[257]={0};//histogram value
	int y[256]={0};
	int area;	//width*height Total pixel
	double p[256]={0.0};// Probability of x[i]
	double mean=0.0 ,var = 0.0 ,sigma=0.0;//mean , covariance 	
	char c[10];//for hisogram coordinate

	/// Initialization
	///********************************************************************************
	area = img.rows*img.cols;//width*height
	imgblur = cv::Mat(img.rows, img.cols, CV_8UC1 );// build a 1 channel image for gray scale image
	imgblur2 = cv::Mat(img.rows, img.cols, CV_8UC1 );
	imgb = cv::Mat::zeros(img.rows, img.cols, CV_8UC1 );
	imgb2 = cv::Mat::zeros(img.rows, img.cols, CV_8UC1 );
	///********************************************************************************

	clock_t start, end;//calculate code processing time
	float total_time = 0;
    start = clock();
	//bilateralFilter(img, imgblur2, 11,20 ,20);
	//adaptiveBilateralFilter(img,imgblur,Size(7,7),100);
	//GaussianBlur( img, imgblur, Size(3,3),0);

	blur(img,imgblur,cv::Size(3,3));
	//medianBlur ( img,imgblur,3);
	for(int i = 0;i < imgblur.rows;i++)
    {
	    for(int j = 0;j < imgblur.cols;j++)
        {
			x[imgblur.at<uchar>(i, j)]++;// calculate the numbers of pixel for 0~255 
        }
    }
	//imshow("img",img);
	//imshow("imgblur",imgblur);
	//imwrite("imgblur_5.bmp",imgblur);
	Mat img1=imread("5.bmp",1); //load image
	Mat imgw = imgblur = cv::Mat(img1.rows, img1.cols, CV_32FC1 );
	watershed(img1, imgw);
	//imshow("imgw",imgw);
	///mean,std
	Scalar mean_scalar,stddev_scalar;//mean and standard deviation
	float mean_x,stddev_x;
    cv::meanStdDev ( imgblur, mean_scalar, stddev_scalar );
    mean_x = mean_scalar.val[0];
    stddev_x = stddev_scalar.val[0];
	printf("mean=%f,std=%f",mean_x,stddev_x);

	int threshold_index = 0;
	int valley=0,valley_index=0;
	int localmin_index=0;
	int x_highpass[257]={0};
	int peakindex[256]={0};
	x[256]=0;
	for(int i=0;i<256;i++)
	{
		x_highpass[i]=x[i+1]-x[i-1];
		x_dif[i] = x[i]-x[i-1];
		x_dif2[i] = x[i]-x[i+1];
	}
	for(int i=0;i<256;i++)
		printf("x[%d]=%d,x_hp[%d]=%d\n",i,x[i],i,x_highpass[i]);
	int peak_number=0;
	///find peak
	for(int i=2;i<200;i++)
	{
		if(x[i]!=0)
		{
			///Peak condition
			if(x_dif[i]>0 && x_dif2[i]>0 && x_dif[i-1]>0 )
			{
				if( abs(x_highpass[i])>0 )
				{
					peakindex[i]=1;
					peak_number++;
				    printf("peak=%d,%d\n",i,x[i]);
				}
			}
		}
	}
	///sort peak
	int *peak = new int[peak_number];
	int *peak2 = new int[peak_number];
	printf("\npeak_number=%d\n",peak_number);
	int count=0;
	for(int i=1;i<200;i++)
	{	
		if(peakindex[i]==1)
		{
			peak[count] = x[i];
			peak2[count] = i;
			count++;
		}		
	}
	bubblesort(peak,peak2,peak_number);
	printf("\n");
	for(int i=0;i<peak_number;i++)
		printf("%d\tpeak=%d\n",peak2[i],peak[i]);
	printf("\nPeak1=%d,Peak2=%d\n",peak2[0],peak2[1]);
	int peak_right = peak2[1];
	///biggest value peak
	if(peak2[0]>peak_right)
		peak_right = peak2[0];
	///find the right peak's valley 
	int valley_right;
	for(int i=peak_right;i<255;i++)
	{
		if(x[i]<x[i+1])
		{
			valley_right = i+1 ;
			break;
		}

	}
	printf("\nvalley_right=%d\n",valley_right);
	///=====================================================================================

	//threshold(img, imgb, valley_right, 255, CV_THRESH_BINARY);
	//imshow("binary",imgb);
	otsu_method(img,imgb2,peak_right);
	imshow("binary2",imgb2);
	//imwrite("otsu1.bmp",imgb2);
	//threshold(img, imgb, 0, 255, CV_THRESH_OTSU);	
	Mat img_morphlogy;
	int m[] = {    1,  1,  1,  1,  1,
                   1,  1,  1,  1,  1,
				   1,  1,  1,  1,  1,
				   1,  1,  1,  1,  1,
				   1,  1,  1,  1,  1
	          };

    Mat kernel = cv::Mat(5,5, CV_8UC1, m); 
	Mat kernel3 = Mat::ones(3,3, CV_8UC1);
	Mat kernel5 = Mat::ones(5,5, CV_8UC1);
	//dilate(imgb2,img_morphlogy,kernel);
	morphologyEx(imgb2,img_morphlogy,MORPH_CLOSE,kernel3,cv::Point(-1,-1),1);
	//erode(img_morphlogy,img_morphlogy,Mat(),cv::Point(-1,-1),2);
	//dilate(img_morphlogy,img_morphlogy,Mat(),cv::Point(-1,-1),2);
	
	imshow("morphlogy",img_morphlogy);
	//imwrite("morph_5x5closing_6.bmp",img_morphlogy);
	///kernel
/*	float m[] =  {0, -1.0, 0,
                   -1.0, 4.0, -1.0,
				   0, -1.0, 0};
    Mat kernel = cv::Mat(3, 3, CV_32FC1, m); 
	Mat out;
    int ddepth = -1;
    cv::filter2D(imgblur, out, ddepth, kernel);
	//imshow("filter",out);
	//imwrite("lap.bmp",out);*/


/*
	///================================Hough transform
	Mat dst, cdst;
    Canny(imgb2, dst, 100, 200, 3);
	imshow("canny", dst);
    cvtColor(dst, cdst, CV_GRAY2BGR);
    vector<Vec4i> lines;
    HoughLinesP(dst, lines, 1, CV_PI/180, 100, 50, 10 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
         Vec4i l = lines[i];
         line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
    }    
    imshow("detected lines", cdst);
	//imwrite("hough.bmp",cdst);

*/
	///===================================== mean var =============================================
	/// mean
/*	for(int i=0;i<256;i++)
    {
        p[i]=(double)x[i]/area;//probability of x[i] (divided by total number of pixel)
		mean = mean + i*p[i];//mean	
		//cout<<i<<" "<<x[i]<<endl;
	}	
	///variance
	for(int i=0;i<256;i++)
    {
		  var = var + pow((i-mean),2)*p[i];
	}
	sigma = pow(var,0.5);//standard deviation*/

	///===================3Sigma+eq+filter+otsu===============
/*	double sigma_3=mean+3*sigma;
    for(int i = 0;i < img.rows;i++)
    {
	     for(int j = 0;j <img.cols;j++)
         {
			 if(img.at<uchar>(i, j)>=sigma_3)
				imgb.at<uchar>(i, j)=255;
		 }
    }*/
	//Mat morph;
	//dilate(imgb,morph,Mat());
	//erode(eroded,eroded,Mat());
	
	///=========================================================================================
	/*int s =0;
	int i;
    #pragma omp parallel for
    for(i = 0; i <= 100; ++i) {
     printf("thread %d : %d\n", omp_get_thread_num(), s = s + i);
    }*/

	///=======================================================================================
	///========================================End Clock====================================== 
	end = clock();//end computing the execution time
    total_time = (float)(end-start)/CLOCKS_PER_SEC;//computing the execution time
    printf("Time : %f sec \n", total_time);//output
	printf("Number of core : %d\n", omp_get_num_procs());
    printf("Number of threads : %d\n", omp_get_num_threads());
	Mat mm = Mat(2, 2, CV_8UC3, Scalar(1,1,1));
    std::cout << mm;

	///========================================Show and Save Image================================
	/*imshow("img",img);
	imshow("binary",imgb);
	imshow("binary2",imgb2);
	imshow("morph",255-morph);*/

	//imwrite("result.bmp",img);
	//imwrite("img_b.bmp",imgb);
	//imwrite("erode.bmp",eroded);
	

	cvWaitKey(0);

}
void prob(Mat& img,double p[256])
{
	int x[256]={0};
	int area;
	area = img.rows*img.cols;

	for(int i = 0;i < img.rows;i++)
    {
	    for(int j = 0;j < img.cols;j++)
        {
			x[img.at<uchar>(i, j)]++;// calculate the numbers of pixel for 0~255 
        }
    }
	for(int i=0;i<256;i++)
    {
        p[i]=(double)x[i]/area;//probability of x[i] (divided by total number of pixel)	
	}	
}
void eq(Mat& imgray, Mat& imgrayeq)
{
	
	 double p[256]={0.0};
	 double p2[256]={0.0};// CDF
	 double py[256]={0.0};
	 int p3[256]={0};//(L-1)*p2[i]
	 prob(imgray,p);
	 for(int i=0;i<256;i++)
     {
		  ///Cumulative distribution function   CDF
		   if(i<1)
		       p2[i]=p[i];//probability
		   else
		   {
		       p2[i]=p2[i-1]+p[i];	         
		   }
		   p3[i]=p2[i]*255+0.5;//S=T(r)

		   for(int j=0;j<256;j++)
           {
			   if(j==p3[i])          	         	         					   
				  py[j]=py[j]+p[i];
			   
		   } 
		 
     }
	 ///Equalization to image
	 for(int i = 0;i < imgray.rows;i++)
     {
	    for(int j = 0;j <imgray.cols;j++)
        {			
			imgrayeq.at<uchar>(i, j)=p3[(int)imgray.at<uchar>(i, j)];
		}
     }

}
void otsu_method(Mat& imgray, Mat imgb,int bound)
{

	double pp1[256]={0.0} ,pp2[256]={0.0},u1[256]={0.0},u2[256]={0.0};
	double th=0.0 ,max_th=0.0;
	int thindex=0;
	double py[256]={0.0};
	prob(imgray,py);
	double uu1=0.0,uu2=0.0;
	///Otsu algorithm

	 for(int t=0; t<bound ; t++)
	 {
		 for(int i=0;i<t+1;i++)
			 pp1[t]=pp1[t] +py[i]; ///sum of probability for 0 ~ t 
		 for(int i=t+1;i<bound;i++)
		     pp2[t]=pp2[t]+py[i]; ///sum of probability for t+1 ~ bound 

		 ///*****************************************************************
		 for(int i=0;i<t+1;i++)
			 u1[t]=u1[t]+py[i]*i/pp1[t]; ///mean
		 
		 for(int j=t+1;j<bound;j++)		 
			 u2[t]=u2[t]+py[j]*j/pp2[t];

		// u1[t] = u1[t]/pp1[t]; ///class mean
		// u2[t] = u2[t]/pp2[t];
		 th=pp1[t]*pp2[t] * pow((u1[t]-u2[t]),2);///otsu
		
		 if(th>max_th)
		 {
			  max_th=th;
		      thindex=t;
		 }///find max value	
		 uu1 = u1[t];
		 uu2 = u2[t];
	}
	cout<<u1[thindex]<<" "<<u2[thindex]<<endl;
	cout<<"otsu threshold="<<thindex<<endl;
	for(int i = 0;i < imgray.rows;i++)
    {
	    for(int j = 0;j <imgray.cols;j++)
        {
			if(imgray.at<uchar>(i, j)>thindex)
				imgb.at<uchar>(i, j)=255;
			else
			    imgb.at<uchar>(i, j)=0;

		}
     }
}
void bubblesort(int *arr,int *index, int n)
{
	//index = new int[n];
    for (int i=0;i<n;i++)
    {
        for (int j=0; j<n; j++)
        {
            if(arr[j]<arr[j+1])
            {
                int temp =  arr[j];//change
                arr[j]=arr[j+1];
                arr[j+1]=temp;
				int temp2 = index[j];//change
                index[j]=index[j+1];
                index[j+1]=temp2;

            }			
        }
     }
}
