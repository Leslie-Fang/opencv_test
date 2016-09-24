#-------------------------------------------------
#
# Project created by QtCreator 2016-09-18T21:59:48
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = test
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

INCLUDEPATH += /usr/local/include \
                /usr/local/include/opencv \
                /usr/local/include/opencv2

LIBS += /usr/lib/x86_64-linux-gnu/libopencv_highgui.so \
 /usr/lib/x86_64-linux-gnu/libopencv_core.so \
 /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so\
/usr/lib/x86_64-linux-gnu/libopencv_ml.so


