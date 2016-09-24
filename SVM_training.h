#ifndef SVM_TRAINING_H
#define SVM_TRAINING_H
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv/cxcore.h"
#include "opencv/ml.h"
#include <iostream>
#include <string.h>


using namespace std;
using namespace cv;

int count_image();
void svm_train(int total_num);

#endif // SVM_TRAINING_H
