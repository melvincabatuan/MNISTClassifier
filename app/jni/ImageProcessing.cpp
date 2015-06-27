/****************************************************************************/
/*

The MIT License (MIT)
 
Copyright (c) Melvin Cabatuan
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

/****************************************************************************/

#include "io_github_melvincabatuan_mnistclassifier_MainActivity.h"
#include <android/log.h>
#include <android/bitmap.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

using cv::Ptr;
using cv::Mat;
using cv::getTickCount;
using cv::getTickFrequency;


#define  LOG_TAG    "MNISTClassifier"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  DEBUG 1


/** Global variables */
Ptr<RTrees> model;
char rtree_path[80];


double t; // for measuring time performance


/*
 * Class:     io_github_melvincabatuan_mnistclassifier_MainActivity
 * Method:    predict
 * Signature: (Landroid/graphics/Bitmap;[B)V
 */
JNIEXPORT void JNICALL Java_io_github_melvincabatuan_mnistclassifier_MainActivity_predict
  (JNIEnv * pEnv, jobject clazz, jobject pTarget, jbyteArray pSource){

   AndroidBitmapInfo bitmapInfo;
   uint32_t* bitmapContent; // Links to Bitmap content

   if(AndroidBitmap_getInfo(pEnv, pTarget, &bitmapInfo) < 0) abort();
   if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) abort();
   if(AndroidBitmap_lockPixels(pEnv, pTarget, (void**)&bitmapContent) < 0) abort();

   /// Access source array data... OK
   jbyte* source = (jbyte*)pEnv->GetPrimitiveArrayCritical(pSource, 0);
   if (source == NULL) abort();

   /// cv::Mat for YUV420sp source and output BGRA 
    Mat srcGray(bitmapInfo.height, bitmapInfo.width, CV_8UC1, (unsigned char *)source);
    Mat mbgra(bitmapInfo.height, bitmapInfo.width, CV_8UC4, (unsigned char *)bitmapContent);

/***********************************************************************************************/
    /// Native Image Processing HERE... 
    if(DEBUG){
      LOGI("Starting native image processing...");
    }

    /// Load RTree classifier (Once ONLY)
    if (model.empty()){ 

       t = static_cast<double>(getTickCount());

       sprintf( rtree_path, "%s/%s", getenv("ASSETDIR"), "RTrees_MNIST.xml");      

       Ptr<RTrees> model = StatModel::load<RTrees>(rtree_path);

       if (model.empty()){ 
           LOGE("Error loading classifier"); 
           abort(); 
       }
 

       t = 1000*(static_cast<double>(getTickCount()) - t)/getTickFrequency();
       if(DEBUG){
          LOGI("Loading classifier took %0.2f ms.", t);
       }
    }
            
  


       //-- Predict number  
       t = static_cast<double>(getTickCount());
     
        

       t = 1000*(static_cast<double>(getTickCount()) - t)/getTickFrequency();
       if(DEBUG){
          LOGI("Predict time = %0.2f ms.", t);
      }


       


       /// Display to Android
       cvtColor(srcGray, mbgra, CV_GRAY2BGRA);


      if(DEBUG){
        LOGI("Successfully finished native image processing...");
      }
   
/************************************************************************************************/ 
   
   /// Release Java byte buffer and unlock backing bitmap
   pEnv-> ReleasePrimitiveArrayCritical(pSource,source,0);
   if (AndroidBitmap_unlockPixels(pEnv, pTarget) < 0) abort();
}
