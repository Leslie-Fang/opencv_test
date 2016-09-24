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

using namespace std;
using namespace cv;

#define calibration_image_area_width 720
#define calibration_image_area_height 540

#define raw_image_area_width_c 720
#define raw_image_area_height_c 540

int main()
{
int camera_seq = 0;
CvCapture * cam = cvCreateCameraCapture(camera_seq);
QTimer *timer;
if(cam==0)
    {
        //bool_open_camera=false;
    }
    else
    {
        //设定捕获图像大小及帧率
        cvSetCaptureProperty(cam,CV_CAP_PROP_FPS,30);
        cvSetCaptureProperty(cam,CV_CAP_PROP_FRAME_WIDTH,raw_image_area_width_c);
        cvSetCaptureProperty(cam,CV_CAP_PROP_FRAME_HEIGHT,raw_image_area_height_c);

        timer->start(33);              // 开始计时，超时则发出timeout()信号，30帧/s
        //bool_open_camera=true;

        unsigned int image_counter = 0;
        unsigned int total_number = 0;
    }


cout<<"hello"<<endl;
return 0;
}
