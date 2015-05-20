#include <opencv2/opencv.hpp> 
#include <opencv2/imgproc/imgproc_c.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h> 
#include <algorithm>
#include <omp.h>

using namespace std;
using namespace cv;

void on_trackbar(int pos);
void prob(Mat& img,double p[256]);
void eq(  Mat& imgray, Mat& imgrayeq);
void otsu_method(Mat& imgray, Mat imgb,int lowbound,int upbound);
void bubblesort(int *arr,int *index, int n);
void drawlabel(cv::Mat &output, std::vector < std::vector<cv::Point2i> > &blobs);
void FindBlobs(const cv::Mat &binary, std::vector <std::vector<cv::Point2i>> &blobs,int intensity);

void FirLowPass(int *arr_in,int *arr_output,int length);
void peak_search(int *x,int* peakindex,int* peak_number,int* peak,int* peak_order);
int mid_peak(int* x,int peaknumber,int* peak,int* peak_order,float left_bound);
int leftpeak_search(float threshold,int* x,int* peakindex );
Mat remove_small(Mat& img,Mat& imgb,int binary,char* str);

Mat img1,img2;

int main()
{
	//load image
	char* filename = "1.bmp";
	Mat img=imread(filename,0); 
	//image
	Mat imgray,imgblur,imgblur2,imgb,result;
	Mat imgb2;
	//histogram
	Mat histogram,equalization,equalization2; 
	int r,g,b,gray;
	//histogram value
	int x[256]={0};
	//histogram value
	int x_dif[257]={0},x_dif2[257]={0};
	int y[256]={0};
	//width*height Total pixel
	int area;	
	// Probability of x[i]
	double p[256]={0.0};
	//mean , covariance 	
	double mean=0.0 ,var = 0.0 ,sigma=0.0;
	//for hisogram coordinate
	char c[10];

	/// Initialization
	///********************************************************************************
	//width*height
	area = img.rows*img.cols;
	// build a 1 channel image for gray scale image
	imgblur = cv::Mat(img.rows, img.cols, CV_8UC1 );
	imgblur2 = cv::Mat(img.rows, img.cols, CV_8UC1 );
	imgb = cv::Mat::zeros(img.rows, img.cols, CV_8UC1 );
	imgb2 = cv::Mat::zeros(img.rows, img.cols, CV_8UC1 );
	///********************************************************************************
	//calculate code processing time
	clock_t start, end;
	float total_time = 0;
    start = clock();
	//bilateralFilter(img, imgblur2, 11,20 ,20);
	//adaptiveBilateralFilter(img,imgblur,Size(7,7),100);
	//GaussianBlur( img, imgblur, Size(3,3),0);
	Mat img_eq = cv::Mat::zeros(img.rows, img.cols, CV_8UC1 );;
	eq(img,img_eq);
	blur(img,imgblur,cv::Size(3,3));
	//imgblur = img;
	//medianBlur ( img,imgblur,3);

	for(int i = 0;i < imgblur.rows;i++)
    {
	    for(int j = 0;j < imgblur.cols;j++)
        {
			// calculate the numbers of pixel for 0~255 
			x[imgblur.at<uchar>(i, j)]++;
        }
    }
	//imshow("imgblur",imgblur);
	int low_bound=0;
	for(int i=0;i<256;i++)
	{
		if(x[i]>0)
		{
			low_bound = i;
			break;
		}
	}
	

	/* mean,std */
	//mean and standard deviation
	Scalar mean_scalar,stddev_scalar;
	float mean_x,stddev_x;
    cv::meanStdDev ( imgblur, mean_scalar, stddev_scalar );
    mean_x = mean_scalar.val[0];
    stddev_x = stddev_scalar.val[0];
	printf("mean=%f,std=%f\n",mean_x,stddev_x);

	int threshold_index = 0;
	int valley=0,valley_index=0;
	int localmin_index=0;
	int x_highpass[257]={0};double x_lowpass[257]={0};
	x[256]=0;
	for(int i=0;i<256;i++)
	{
		x_highpass[i]=x[i+1]-x[i-1];
		x_dif[i] = x[i]-x[i-1];
		x_dif2[i] = x[i]-x[i+1];
	}
	
///=================================================
///   Find all peaks 
///=================================================
	/*FIR Low pass filter*/
	int lp_x[256]={0};
	int FIR_seq = 10;
	FirLowPass(x,lp_x,FIR_seq);

	int peak_number=0;
	int peakindex[256]={0};
	int peak[256]={0};
	int peak_order[256]={0};
	peak_search(lp_x,peakindex,&peak_number,peak,peak_order);
/*	for(int i=0;i<256;i++)
		printf("%d\tlp_x=%d\n",i,lp_x[i]);*/
	for(int i=0;i<peak_number;i++)
		printf("%d\tpeak=%d\n",peak_order[i],peak[i]);
///=================================================
///  find peak on the 'Left side'
///=================================================
	int peak_diewall_index = leftpeak_search(mean_x-stddev_x,x,peakindex );

///=================================================
///   the biggest peaks 
///=================================================
	int midpeak = mid_peak(x,peak_number,peak,peak_order,mean_x-stddev_x);
	printf("midpeak=%d",midpeak);
	otsu_method(imgblur,imgb,0,midpeak);
	medianBlur(imgb,imgb,5);
	imshow("otsu LEFT",imgb);
	
///=====================================================
///               find rightest peak's valley 
///=====================================================
	int rightest_peak=255;
	for(int i=0;i<peak_number;i++)
	{
		if(peak_order[i]>rightest_peak)
			rightest_peak = peak_order[i];
	}
	int rightest_peak_v = 0;
	for(int i=rightest_peak ;i>0;i--)
	{
		if(x[i] < x[i-1])
		{
			rightest_peak_v = i;
			break;
		}				
	}
	printf("rightest_peak = %d\nrightest_peak_v=%d\t",rightest_peak,rightest_peak_v);
	/*threshold(imgblur, imgb, 225, 255, CV_THRESH_BINARY); 
	medianBlur(imgb,imgb,5);
	imshow("otsu right",imgb);*/
///=====================================================
///                  track bar threshold
///=====================================================
/*	img1 = imgblur.clone();
	img2 = cv::Mat::zeros(img.rows, img.cols, CV_8UC1 );
	namedWindow("binary", CV_WINDOW_AUTOSIZE);
    int nThreshold = 0;  
    cvCreateTrackbar("binarization","binary", &nThreshold, 254, on_trackbar);  
    on_trackbar(0); */
///=====================================================
///                  image binarization
///===================================================== 
    Mat imgb_2,imgb_3;
	Mat imgb_combine;
	//threshold(imgblur, imgb, 50, 255, CV_THRESH_BINARY); 
	//threshold(imgblur, imgb2, 250, 255, CV_THRESH_BINARY); 
	imgb_combine = (255-imgb)+imgb2;
	cout<<"mean_x="<<mean_x<<endl;
	otsu_method(imgblur,imgb,peak_diewall_index,mean_x);
	medianBlur(imgb,imgb,5);
	///imshow("otsu-left2",imgb);

///=====================================================
///                  remove small object noises
///=====================================================		
	Mat img_denoise = remove_small(img,imgb,0,"b");
	Mat connected_label = remove_small(imgblur,img_denoise,255,"rgb");
	imshow("img_denoise",img_denoise);
	imshow("connected_label",connected_label);
	//imwrite("compensate6.bmp",img_denoise);
	//Mat imgb_inv = 255-img_denoise;
	//imshow("otsu imgb_inv",imgb_inv);
///=====================================================
///           mapping to original image
///=====================================================
	Mat img_roi = cv::Mat::zeros(img.rows, img.cols, CV_8UC1 );
	for(int i = 0;i < connected_label.rows;i++)
    {
	     for(int j = 0;j <connected_label.cols;j++)
         {
			 if(connected_label.at<cv::Vec3b>(i,j) == cv::Vec3b(0,0,0))
				 img_roi.at<uchar>(i, j)=img.at<uchar>(i, j);
			 else
				 img_roi.at<uchar>(i, j)=0;
		 }
    }
	imshow("img_roi",img_roi);
	//need space for string behind
	char write_name[20] = "img_roi";
	strcat(write_name,filename);
	//imwrite(write_name,img_roi);


///=====================================================
///           compensate for broken line
///=====================================================
	int window = 3;
	for(int i = (window-1)/2;i < imgb.rows-(window-1)/2;i++)
    {
	     for(int j = (window-1)/2;j <imgb.cols-(window-1)/2;j++)
         {
			 if(img_denoise.at<uchar>(i, j)==0)
			 {
				 for(int i2=i-(window-1)/2 ;i2<=i+(window-1)/2 ;i2++)
				 {
					  for(int j2=j-(window-1)/2; j2<=j+(window-1)/2 ;j2++)
					  {
						  //tune
						  if( img.at<uchar>(i2,j2)<0.5*midpeak)
							  img_denoise.at<uchar>(i2,j2)=0;
					  }
				 }			 				  
			 }			
		 }
    }
	///imshow("edge compensate",img_denoise);
	//imwrite("compensate1.bmp",imgb);
///=====================================================
///                  image morphlogy
///=====================================================				
/*	Mat img_morphlogy;
	///not suitable for all direction 
	int m[] = { 
		           0,  1,  0,  
                   1,  1,  1,  
				   0,  1,  0, 
	          };

    Mat kernel = cv::Mat(3,3, CV_8UC1, m); 
	Mat kernel_3 = Mat::ones(3,3, CV_8UC1);
	Mat kernel_5 = Mat::ones(5,5, CV_8UC1);
	dilate(img_denoise,img_denoise,kernel);
	morphologyEx(img_denoise,img_morphlogy,MORPH_OPEN,kernel_3,cv::Point(-1,-1),1);
	morphologyEx(img_morphlogy,img_morphlogy,MORPH_OPEN,kernel_3,cv::Point(-1,-1),1);
///	imshow("morphlogy",img_morphlogy);
	//imwrite("morph1.bmp",img_morphlogy);

	std::vector<std::vector<cv::Point> > contours;
	vector<Vec4i> hierarchy;
    cv::Mat contourOutput = 255-img_morphlogy;
	Mat contourImage = Mat::zeros( img_morphlogy.size(), img_morphlogy.type() );;
    cv::findContours( contourOutput, contours,hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE );
	cv::drawContours(contourImage, contours, -1, 255,1,8,hierarchy,1);*/
	//imshow("contourImage",contourImage);

	//imwrite("morph6.bmp",img_morphlogy);
	///connected component labeling
/*	std::vector < std::vector<cv::Point2i > > blobs2;
	Mat connected_label2 = cv::Mat::zeros(img.size(), CV_8UC3);
	FindBlobs(img_morphlogy,blobs2);//find connected component
	drawlabel(connected_label2,blobs2);
	for(int i=0;i<blobs2.size();i++)
		printf("size of %d=%d\n",i,blobs2[i].size());
	imshow("connected  2",connected_label2);*/

	//imwrite("connected6.bmp",connected_label);

///=====================================================
///           mapping to original image
///=====================================================
/*
	//Mat img_morph_inv = 255-img_morphlogy;
	Mat img_and = cv::Mat::zeros(img.rows, img.cols, CV_8UC1 );
	/// AND operation
	for(int i = 0;i < img_morphlogy.rows;i++)
    {
	     for(int j = 0;j <img_morphlogy.cols;j++)
         {
			 if(img_morphlogy.at<uchar>(i, j)==255)
				 img_and.at<uchar>(i, j)=img.at<uchar>(i, j);
			 else
				 img_and.at<uchar>(i, j)=0;
		 }
    }
	imshow("img_and",img_and);*/
	//imwrite("img_and6.bmp",img_and);

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
	system("pause");
	return 0;

}
void on_trackbar(int pos)  
{  
    cv::threshold(img1, img2, pos, 255, CV_THRESH_BINARY);   
    imshow("binary", img2);  
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
void otsu_method(Mat& imgray, Mat imgb,int lowbound,int upbound)
{

	double pp1[256]={0.0} ,pp2[256]={0.0},u1[256]={0.0},u2[256]={0.0};
	double th=0.0 ,max_th=0.0;
	int thindex=0;
	double py[256]={0.0};
	prob(imgray,py);
	double uu1=0.0,uu2=0.0;
	///Otsu algorithm

	 for(int t=lowbound; t<upbound ; t++)
	 {
		 for(int i=lowbound;i<t+1;i++)
			 pp1[t]=pp1[t] +py[i]; ///sum of probability for 0 ~ t 
		 for(int i=t+1;i<upbound;i++)
		     pp2[t]=pp2[t]+py[i]; ///sum of probability for t+1 ~ bound 

		 ///*****************************************************************
		 for(int i=lowbound;i<t+1;i++)
			 u1[t]=u1[t]+py[i]*i/pp1[t]; ///mean
		 
		 for(int j=t+1;j<upbound;j++)		 
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
void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs,int intensity)
{
    blobs.clear();
    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground (before label)
    // 2+ - labelled foreground

    cv::Mat label_image;
    binary.convertTo(label_image, CV_32SC1);//32bits image

    int label_count = 2; // pixel value , starting at 2 because 0(black),1(white) are already used
	//New value of the repainted domain pixels
    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);///
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != intensity) {
                continue;
            }

            cv::Rect rect;//rectangle
			//connected component
            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);

            std::vector <cv::Point2i> blob;//new blobs

			//for same label value
            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);///
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }

                    blob.push_back(cv::Point2i(j,i));//same label value ,push to same vector
                }
            }

            blobs.push_back(blob);//number of blobs ++
            label_count++;//next label value
        }
    }
}
///components labeling
void drawlabel(cv::Mat &output, std::vector < std::vector<cv::Point2i> > &blobs)
{
	// Random coloring the blobs with rgb color
    for(size_t i=0; i < blobs.size(); i++) 
	{
        unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));

        for(size_t j=0; j < blobs[i].size(); j++) 
		{
            int x = blobs[i][j].x;//position
            int y = blobs[i][j].y;

            output.at<cv::Vec3b>(y,x)[0] = b;//assign color to image
            output.at<cv::Vec3b>(y,x)[1] = g;
            output.at<cv::Vec3b>(y,x)[2] = r;
        }
    }
}
void FirLowPass(int *arr_in,int *lp_arr_output,int length)
{
	for(int i=0;i<256;i++)
	{
		lp_arr_output[i] = arr_in[i];
		for(int j=0;j<=length;j++)
		{
			if(i-j>0)
				lp_arr_output[i] += (double)arr_in[i-j];
			else
				lp_arr_output[i]= lp_arr_output[i];
		}
		lp_arr_output[i] = (int)(lp_arr_output[i] / length);
	}
	
}
void peak_search(int *x,int* peakindex,int* peak_number,int* peak,int* peak_order)
{
	//histogram value
	int x_highpass[257]={0},x_dif[257]={0},x_dif2[257]={0};
	int n=0;
	for(int i=0;i<256;i++)
	{
		x_highpass[i]=x[i+1]-x[i-1];
		x_dif[i] = x[i]-x[i-1];
		x_dif2[i] = x[i]-x[i+1];
	}
	for(int i=1;i<256;i++)
	{
		if(x[i]!=0)
		{
			/* Peak condition */
			if(x_dif[i]>0 && x_dif2[i]>0)// && x_dif2[i+1]>0  )
			{
				peakindex[i]=1;
				n++;
			}
		}
	}
	*peak_number = n;

	int count=0;
	for(int i=1;i<256;i++)
	{	
		if(peakindex[i]==1)
		{
			peak[count] = x[i];
			peak_order[count] = i;
			count++;
		}		
	}
	/* peak value sorting*/
	bubblesort(peak,peak_order,n);	
}

int mid_peak(int* x,int peaknumber,int* peak,int* peak_order,float left_bound)
{
	/* 2 biggest peaks */
	printf("\nPeak1=%d,Peak2=%d\n",peak_order[0],peak_order[1]);
	int peak_right = peak_order[1];
	int peak_left = peak_order[0];
	/* biggest value peak */
	if(peak_left>peak_right)
	{
		peak_right = peak_left;
		peak_left = peak_order[1];
	}
	/* find the right peak's valley */
	//int valley_right,valley_left_right;
	int mid_peak_left=peak_right;
	for(int i=0;i<peaknumber;i++)
	{
		if(peak_order[i]>left_bound && peak_order[i]<peak_right)
		{
			if(peak_order[i] < mid_peak_left)
			{
				mid_peak_left = peak_order[i];
			}
		}
	
	}
	
	printf("\nvalley_right=%d\n",mid_peak_left);
	return mid_peak_left;

	
}
/* mean_x-stddev_x */
int leftpeak_search(float threshold,int* x,int* peakindex )
{
	//histogram value
	int x_highpass[257]={0},x_dif[257]={0},x_dif2[257]={0};
	for(int i=0;i<256;i++)
	{
		x_highpass[i]=x[i+1]-x[i-1];
		x_dif[i] = x[i]-x[i-1];
		x_dif2[i] = x[i]-x[i+1];
	}
	int peak_diewall = 0,peak_diewall_index=0;
	for(int i=2;i<=threshold ;i++)
	{
		if( peakindex[i] == 1 )
		{
			/* Peak condition */		
			if(x_dif[i]>0 && x_dif2[i]>0 && x_dif[i-1]>0 && x_dif2[i+1]>0  )
			{
				 ///printf("peak=%d,%d\tx_highpass[%d]=%d\n",i,x[i],i+1,x_highpass[i+1]);
				 /* High negative value of peak */
				 if(x_highpass[i+1] < peak_diewall)
				 {
					 peak_diewall = x_highpass[i+1] ;
					 peak_diewall_index = i;
				 }	 
			}
		}
	}
	/* decide whether the peak_diewall_index has big intensity */
	for(int i=2;i<=threshold ;i++)
	{
		if( peakindex[i] == 1 )
		{
			if(peak_diewall_index > i)
			{
				if( x[i] > x[peak_diewall_index] )
				{
					peak_diewall_index = i;
				}
			}			
		}
	}
	//printf("peak_diewall_index=%d\n",peak_diewall_index);
	return peak_diewall_index;
}
Mat remove_small(Mat& img,Mat& imgb,int binary,char* str)
{
	std::vector < std::vector<cv::Point2i > > blobs;
	Mat connected_label = cv::Mat::zeros(imgb.size(), CV_8UC3);
	// find connected component
	FindBlobs(imgb,blobs,binary);
	// draw rgb color on labels
	drawlabel(connected_label,blobs);
	// sum of total size of object
    int sum_component_size = 0;
	for(int i=0;i<blobs.size();i++)
	{
		//printf("size of %d=%d\n",i,blobs[i].size());
		sum_component_size+=blobs[i].size();
	}
	printf("sum=%d",sum_component_size);

	Mat remove_small = imgb.clone();

	if(str=="rgb")
	{
		for(size_t i=0;i<blobs.size();i++)
	    {
		    if(blobs[i].size()<(int)(0.5*sum_component_size/blobs.size())  )
		    {
				int region_sum_intensity = 0;
				double avg_intensity=0;

			    for(size_t j=0;j<blobs[i].size();j++)
	            {
				    int x = blobs[i][j].x;
				    int y = blobs[i][j].y;  
					region_sum_intensity += img.at<uchar>(y,x);				
			    }
				avg_intensity = (double)(region_sum_intensity/blobs[i].size());
				printf("region_sum_intensity=%d,avg_intensity=%f\n",region_sum_intensity,avg_intensity);
				///need tune
				if(avg_intensity >160)
				{
					for(size_t j=0;j<blobs[i].size();j++)
	                {
				        int x = blobs[i][j].x;
				        int y = blobs[i][j].y;  				
				        connected_label.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,0);
			        }
				}

		    }			
	    }
		return connected_label;
	}
	    
	else if(str=="b")
	{
		for(size_t i=0;i<blobs.size();i++)
	    {
		    if(blobs[i].size()<(int)(0.5*sum_component_size/blobs.size()) )
		    {
			    for(size_t j=0;j<blobs[i].size();j++)
	            {
				    int x = blobs[i][j].x;
				    int y = blobs[i][j].y;
				    remove_small.at<uchar>(y,x) = 255;				 
			    }
		    }			
	    }
		return remove_small;
	}
		
}
	
