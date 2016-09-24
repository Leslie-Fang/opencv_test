#include "mainwindow.h"
#include <QApplication>
#include <iostream>
#include <string.h>
#include <QWidget>
#include <QImage>
#include <QTimer>
#include <QTime>
#include <QDateTime>
#include <QDir>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv/cxcore.h"
#include "opencv/ml.h"
#include <stdio.h>

using namespace std;
using namespace cv;

#define calibration_image_area_width 720
#define calibration_image_area_height 540

#define raw_image_area_width_c 720
#define raw_image_area_height_c 540

#define CV_RED cvScalar(255,0,0,0)
#define CV_WHITE cvScalar(255,255,255,0)

#define SQUARE_IMAGE_SIZE_R 16
#define SQUARE_IMAGE_SIZE_C 32

#define THRESHOLD 80
#define CLASS_SUM 10

CvSVM svm_classifier;
bool img_right = false;

void rawimage_process();
int count_image();
void svm_train(int total_num);

//处理单张图片
Mat image_process(Mat img_white);
int test_svm();


int main(int argc, char *argv[])
{
   // QApplication a(argc, argv);
   // MainWindow w;
   // w.show();

    //这个函数是对图片进行预处理的
  // rawimage_process();

    //这两个函数得到训练SVM得到的参数，保存在一个文件里面
 //  int n=count_image();
 //  svm_train(n);

    //这个函数验证训练的效果
    //读取一张图片，SVM分类，输出分类的结果
  // test_svm();

//    CvCapture *capture=cvCreateCameraCapture(0);
    IplImage* frame;
    CvCapture* capture = cvCreateCameraCapture(0);
    cvNamedWindow("win");
    for(int i=0;i<10;i++)
    {
        frame = cvQueryFrame(capture);
        if(!frame) break;
        cvShowImage("win", frame);

        char c = cvWaitKey(0);
        if(c==27) continue;

    }
    cvReleaseCapture(&capture);
    cvDestroyWindow("win");
    cout<<"hello"<<endl;
    return 0;

   // cout<<"hello"<<endl;
   // return a.exec();
}

void rawimage_process()
{
    for(int class_seq=0; class_seq<CLASS_SUM; class_seq++)
     {
         for(int img_seq=0; ;img_seq++)
         {
             char base_loadpath[100]="/home/leslie/test_code/qtcreator_test/test/raw/";
             char base_savepath[100]="/home/leslie/test_code/qtcreator_test/test/process/";


             char class_num[4];
             sprintf(class_num,"%d/",class_seq);
             strcat(base_loadpath,class_num);

             char img_num[8];
             sprintf(img_num,"%d.jpg",img_seq);
             strcat(base_loadpath,img_num);

             strcat(base_savepath,class_num);
             strcat(base_savepath,img_num);

             cout<<"begin to read!"<<endl;
             Mat image=imread(base_loadpath,CV_LOAD_IMAGE_COLOR );
             if(image.empty())
             {
                 cout<<"This class is finished! Beginnext class"<<endl;
                // image.release();
                 break;
             }
             cout<<"read successfully ";
             cout<<base_loadpath<<endl;
             //cout<<image<<endl;
             cout<<sizeof(image)<<endl;
             cout<<image.channels()<<endl;

             Mat image_gray;
             Mat image_gray_blur;
             Mat image_binary,image_binary_copy,image_binary_copy2;
             //将图像二值化，遍历所有的点
             //首先将彩色图像转化成灰度图像
             cvtColor(image,image_gray,CV_BGR2GRAY);
             //对图像做平滑处理
             blur( image_gray, image_gray_blur, Size(3,3),Point(-1,-1));
             //二值化
             threshold(image_gray_blur, image_binary, 100, 255, THRESH_BINARY_INV);


             namedWindow("img");
             imshow("img", image);

             namedWindow("img_binary");
             imshow("img_binary", image_binary);

             //开环处理，先膨胀，再腐蚀
             //定义一个膨胀腐蚀需要的像素区间
             vector<vector<Point> > contours;
             Mat element = getStructuringElement( 0,Size( 10 ,10));

             for(int i=0;;i++)
             {
                 dilate(image_binary, image_binary, element);
                 erode(image_binary, image_binary, element);

                 image_binary_copy = image_binary.clone();
                 image_binary_copy2 = image_binary.clone();

                 // Find and draw contours
                 findContours(image_binary_copy2, contours, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
                 cout<<"contours.size()="<<contours.size()<<endl;
                 if(contours.size()==1) break;
                 if(contours.size()==0 || i==4)
                 {
                    cout<<"can not find contour of Class "<<class_seq<<", Image "<<img_seq<<"!"<<endl;
                    break;
                 }
             }
             if(contours.size() != 1)
             {
                 cvWaitKey(10);
                 continue;
             }
             drawContours(image_binary_copy2, contours, 0, CV_RED, 2);

             vector<Rect> boundRect(contours.size());
             //boundingRect 这一步外接矩形的各个点已经生成了
             boundRect[0] = boundingRect(Mat(contours[0]));
             //get the 外接矩形, image_binary_copy need to be initialized to input the size before
             rectangle(image_binary_copy, boundRect[0].tl(), boundRect[0].br(), CV_WHITE, 0.2, 8, 0);

             //根据外接矩形 裁剪并保存图片
             //生成一个矩形框roi(region of interest) 感兴趣的区域
             CvRect roi = CvRect(boundRect[0]);
             IplImage imgage_roi = image_binary;
             //完成裁剪，传入要裁剪的图片的地址，改变指针的位置
             cvSetImageROI(&imgage_roi,roi);

             // Resize,需要保存的图片保存成统一的像素大小
             IplImage* img_final = cvCreateImage(cvSize(SQUARE_IMAGE_SIZE_R,SQUARE_IMAGE_SIZE_C), IPL_DEPTH_8U, 1);
             cvResize(&imgage_roi, img_final, CV_INTER_AREA);


             namedWindow("img");
             imshow("img", image);

             namedWindow("img_binary");
             imshow("img_binary", image_binary);

             cvNamedWindow("img_final", CV_WINDOW_AUTOSIZE);
             cvShowImage("img_final",img_final);

             cvSaveImage(base_savepath, img_final);
             cvWaitKey(10);

             image.release();
             image_gray.release();


             image_gray_blur.release();
             image_binary.release();
             image_binary_copy.release();
             image_binary_copy2.release();
             cvReleaseImage(&img_final);
            // cvReleaseImage(&img_final);
         }
     }
}





int test_svm()
{
    int res=0;
    CvSVM svm;
    svm.load("/home/leslie/test_code/qtcreator_test/test/SVM_DATA.xml");
    cout<<"Finish loading params!"<<endl;
    /*
    char base_loadpath[100]="/home/leslie/test_code/qtcreator_test/test/raw/7/0.jpg";
    Mat image1 = imread(base_loadpath,CV_LOAD_IMAGE_COLOR );
    if(image1.empty())
    {
        cout<<"load failed!"<<endl;
        return 0;
    }
    image1=image_process(image1);
    */
    char base_loadpath[100]="/home/leslie/test_code/qtcreator_test/test/process/1/2.jpg";
    Mat image1 = imread(base_loadpath,CV_LOAD_IMAGE_COLOR );
    if(image1.empty())
    {
        cout<<"load failed!"<<endl;
        return 0;
    }

    namedWindow("im");
    imshow("im",image1);
    cvWaitKey(5000);
    const int rols = 1;
    const int cols = SQUARE_IMAGE_SIZE_R*SQUARE_IMAGE_SIZE_C/4;

     float training_data1[cols];
    int ptr_num = 0;
    for (int j=0; j<SQUARE_IMAGE_SIZE_R; j+=2)   //rols
    {
        for (int i=0; i<SQUARE_IMAGE_SIZE_C; i+=2)  //cols
        {
            float temp = (image1.at<uchar>(i,j)+image1.at<uchar>(i+1,j)+image1.at<uchar>(i,j+1)+image1.at<uchar>(i+1,j+1))/4.0/255.0;
            training_data1[ptr_num] = temp;
            ptr_num ++;
        }
    }

    Mat training_data_mat1(rols, cols, CV_32FC1, training_data1);
    float response1 = svm.predict(training_data_mat1);
    training_data_mat1.release();
    cout<<"response1="<<response1<<endl;

    int response1_n = (int)response1;
    cout<<"response1_n="<<response1_n<<endl;
    if(response1-response1_n < 0.05)
    {
       res=response1_n;
    }
    else if(response1_n+1-response1 < 0.05)
    {
       res=response1_n+1;
    }
    else
    {
       cout<<"Not Sure! "<<endl;
       res= 14;
    }
    return res;
}

//处理单张图片
Mat image_process(Mat image)
{
    if(image.empty())
    {
        cout<<"This class is finished! Beginnext class"<<endl;
       // image.release();
        return image;
    }
    cout<<"read successfully ";
    //cout<<base_loadpath<<endl;
    //cout<<image<<endl;
    cout<<sizeof(image)<<endl;
    cout<<image.channels()<<endl;

    Mat image_gray;
    Mat image_gray_blur;
    Mat image_binary,image_binary_copy,image_binary_copy2;
    //将图像二值化，遍历所有的点
    //首先将彩色图像转化成灰度图像
    cvtColor(image,image_gray,CV_BGR2GRAY);
    //对图像做平滑处理
    blur( image_gray, image_gray_blur, Size(3,3),Point(-1,-1));
    //二值化
    threshold(image_gray_blur, image_binary, 100, 255, THRESH_BINARY_INV);

    //开环处理，先膨胀，再腐蚀
    //定义一个膨胀腐蚀需要的像素区间
    vector<vector<Point> > contours;
    Mat element = getStructuringElement( 0,Size( 10 ,10));

    for(int i=0;;i++)
    {
        dilate(image_binary, image_binary, element);
        erode(image_binary, image_binary, element);

        image_binary_copy = image_binary.clone();
        image_binary_copy2 = image_binary.clone();

        // Find and draw contours
        findContours(image_binary_copy2, contours, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        cout<<"contours.size()="<<contours.size()<<endl;
        if(contours.size()==1) break;
        if(contours.size()==0 || i==4)
        {
           //cout<<"can not find contour of Class "<<class_seq<<", Image "<<img_seq<<"!"<<endl;
           break;
        }
    }
    if(contours.size() != 1)
    {
        cvWaitKey(10);
        //continue;
    }
    drawContours(image_binary_copy2, contours, 0, CV_RED, 2);

    vector<Rect> boundRect(contours.size());
    //boundingRect 这一步外接矩形的各个点已经生成了
    boundRect[0] = boundingRect(Mat(contours[0]));
    //get the 外接矩形, image_binary_copy need to be initialized to input the size before
    rectangle(image_binary_copy, boundRect[0].tl(), boundRect[0].br(), CV_WHITE, 0.2, 8, 0);

    //根据外接矩形 裁剪并保存图片
    //生成一个矩形框roi(region of interest) 感兴趣的区域
    CvRect roi = CvRect(boundRect[0]);
    IplImage imgage_roi = image_binary;
    //完成裁剪，传入要裁剪的图片的地址，改变指针的位置
    cvSetImageROI(&imgage_roi,roi);

    // Resize,需要保存的图片保存成统一的像素大小
    IplImage* img_final = cvCreateImage(cvSize(SQUARE_IMAGE_SIZE_R,SQUARE_IMAGE_SIZE_C), IPL_DEPTH_8U, 1);
    cvResize(&imgage_roi, img_final, CV_INTER_AREA);

 //   cvSaveImage(base_savepath, img_final);
    cvWaitKey(10);

    image.release();
    image_gray.release();
    image_gray_blur.release();
    image_binary.release();
    image_binary_copy.release();
    image_binary_copy2.release();

    return img_final;
}

int count_image()
{
    /**counter**/
        int total_num = 0;

        for(int class_seq=0; class_seq<CLASS_SUM; class_seq++)
        {
            for(int img_seq=0; ;img_seq++)
            {
                char image_name[150] = "/home/leslie/test_code/qtcreator_test/test/process/";
                //add read path and name
                char class_num[4];
                sprintf(class_num,"%d/",class_seq);
                strcat(image_name,class_num);

                char img_num[10];
                sprintf(img_num,"%d.jpg",img_seq);
                strcat(image_name,img_num);

                //read image
                Mat img_training = imread(image_name);

                if(img_training.empty())
                {
                    //cout<<"Found "<<img_seq<<" images in Class "<<class_seq<<". Try next class."<<endl;
                    if(img_seq > 12)
                    {
                        break;
                    }
                    continue;
                }
                else total_num++;

            }
        }
        cout<<"Found "<<total_num<<" images!"<<endl;
        return total_num;

}

void svm_train(int total_num)
{
        /*Load Training Data*/
        int seq_counter = 0;
        cout<<"total_num="<<total_num<<endl;

        //每一张图片一行
        const int rols = total_num;
        const int cols = SQUARE_IMAGE_SIZE_R*SQUARE_IMAGE_SIZE_C/4;
        //cout<<rols<<"  "<<cols<<endl;

        float training_data[rols][cols];
        float training_label[rols];
        for(int class_seq=0; class_seq<CLASS_SUM; class_seq++)
            {
                for(int img_seq=0; ;img_seq++)
                {
                    char image_name[150] = "/home/leslie/test_code/qtcreator_test/test/process/";
                    //add read path and name
                    char class_num[4];
                    sprintf(class_num,"%d/",class_seq);
                    strcat(image_name,class_num);

                    char img_num[10];
                    sprintf(img_num,"%d.jpg",img_seq);
                    strcat(image_name,img_num);
                    cout<<image_name<<endl;

                    //read image
                    Mat img_training = imread(image_name);
                    if(img_training.empty())
                    {
                        cout<<"Processed "<<img_seq<<" images in Class "<<class_seq<<". Start next class."<<endl;
                        if(img_seq > 12)
                        {
                            break;
                        }
                        continue;
                    }
                    else
                    {
                        cout<<"Found "<<img_seq<<endl;
                    }

                   // threshold(img_training, img_training, 80, 255, THRESH_BINARY_INV);
                    //cout<<
                    training_label[seq_counter] = (float)class_seq;
                    cout<<training_label[seq_counter]<<endl;
                    //将所有像素点的值赋值给到training_data

                    int ptr_num = 0;
                    for (int j=0; j<SQUARE_IMAGE_SIZE_R; j+=2)   //rols
                     {
                       for (int i=0; i<SQUARE_IMAGE_SIZE_C; i+=2)  //cols
                       {
                        training_data[seq_counter][ptr_num] = (img_training.at<uchar>(i,j)+img_training.at<uchar>(i+1,j)+img_training.at<uchar>(i,j+1)+img_training.at<uchar>(i+1,j+1))/4.0/255.0;
                        cout<<training_data[seq_counter][ptr_num]<<"  ";
                        ptr_num ++;
                       }
                    }



                    img_training.release();

                    seq_counter++;

                }
        }
        cout<<"Finish reading data...Begin Training..."<<endl;

        /*Start Training*/
        CvSVMParams params;
        params.svm_type    = CvSVM::C_SVC;
        params.kernel_type = CvSVM::LINEAR;
        params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
        cout<<"Parameters Settled!"<<endl;

        Mat labels_mat(rols, 1, CV_32FC1, training_label);
        Mat training_data_mat(rols, cols, CV_32FC1, training_data);

        //cout<<labels_mat<<endl;
        //cout<<training_data_mat<<endl;

        svm_classifier.train(training_data_mat, labels_mat, Mat(), Mat(), params);
        svm_classifier.save( "/home/leslie/test_code/qtcreator_test/test/SVM_DATA.xml" );
        cout<<"Training data saved"<<endl;

        cout<<"Training finished!~"<<endl;

}
