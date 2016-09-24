#include "SVM_training.h"
#include "main.h"

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
                    break;
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
        //cout<<"total_num="<<total_num<<endl;

        //每一张图片一行
        const int rols = total_num;
        const int cols = SQUARE_IMAGE_SIZE_R*SQUARE_IMAGE_SIZE_C;
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
                    Mat img_training = imread(image_name,CV_LOAD_IMAGE_GRAYSCALE);
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

                    threshold(img_training, img_training, 80, 255, THRESH_BINARY_INV);
                    training_label[seq_counter] = (float)class_seq;
                    //将所有像素点的值赋值给到training_data
                    for (int j=0; j<SQUARE_IMAGE_SIZE_R; j++)   //rols
                    {
                                uchar* raw= img_training.ptr<uchar>(j);
                                for (int i=0; i<SQUARE_IMAGE_SIZE_C; i++)  //cols
                                {
                                    training_data[seq_counter][j*SQUARE_IMAGE_SIZE_R+i] = (float)raw[i];
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
