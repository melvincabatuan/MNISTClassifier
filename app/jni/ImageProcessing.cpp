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
#include <string> 
 

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv::ml;

using cv::getTickCount;
using cv::getTickFrequency;
using cv::Mat;
using cv::Ptr;
using cv::Point;
using cv::Rect;
using cv::Scalar;
using cv::Size;

// std
using std::vector;
using std::string;


#define  LOG_TAG    "MNISTClassifier"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  DEBUG 1


/** Global variables */
char rtree_path[80];
Ptr<RTrees> model;
bool isClassifierLoaded = false;
Mat *pScaledDown = NULL;
Mat *pScaledUp = NULL;
Mat *pOtsu = NULL;
cv::RNG rng(12345);


double t; // for measuring time performance



bool isCentral(Point center, float &r, int w, int h){
   return ( ((center.x - r) >= 0) && ((center.y - r) >= 4) && ((center.x + r) <=  w) && ((center.y + r) <= h) );
}




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
      // LOGI("srcGray.size() = (%d, %d)", srcGray.size().height , srcGray.size().width );  // xperia lt15i: srcGray.size() = (288, 352)
    }


     if(pScaledDown == NULL)
       pScaledDown = new Mat(bitmapInfo.height/2, bitmapInfo.width/2, srcGray.type());
 
     Mat scaledDown = *pScaledDown;
     pyrDown( srcGray, scaledDown);



    /// 1. Load RTree classifier (Once ONLY)
    if (!model){ 

       t = static_cast<double>(getTickCount());

       sprintf( rtree_path, "%s/%s", getenv("ASSETDIR"), "rtreesmnist.xml");      

       model = StatModel::load<RTrees>(rtree_path);

       if (!model){ 
           LOGE("Error loading classifier"); 
           abort(); 
       }
 
       t = 1000*(static_cast<double>(getTickCount()) - t)/getTickFrequency();

       if(DEBUG){
          LOGI("Loading classifier took %0.2f ms.", t);
       }

    }
           
 

   /// Classifier successfully loaded!

   /// 2. Otsu Thresholding
 
   if(pOtsu == NULL)
       pOtsu = new Mat(srcGray.size()/2, srcGray.type()/2);
 
    Mat otsu = *pOtsu;
    threshold( scaledDown, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU ); 

    


   /// 3. Detect Contours
     vector< vector<cv::Point> > contours; // detected contours
     findContours( otsu, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));  





   /// if contours are found, update roi rectangle
     if(contours.size() > 0){

         /// Approximate contours to polygons + get bounding circles
         vector< vector<cv::Point> > contours_poly( contours.size() );
         vector<cv::Point2f>center( contours.size() );
         vector<float>radius( contours.size() );

         for( int i = 0; i < contours.size(); i++ )
         { 
             approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
             minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
         }       

         for( int i = 0; i < contours.size(); i++ )
         {
    
             if(radius[i] > 10  && isCentral(center[i], radius[i], scaledDown.cols, scaledDown.rows) ){   
     
               Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
               circle( scaledDown, center[i], (int)radius[i], color, 2, 8, 0 );



               // crop image roi
               Rect r = Rect(Point(center[i].x - (int)radius[i], center[i].y - (int)radius[i]), Point(center[i].x + (int)radius[i], center[i].y + (int)radius[i]));
               Mat ROI = scaledDown(r);


               // resize to 20 x 20
               Mat tmp1, tmp2;

               resize(ROI,tmp1, Size(28,28), 0, 0, cv::INTER_LINEAR );

               /// Convert to float
               tmp1.convertTo(tmp2,CV_32FC1); 

               /// Reshape into row
               Mat row = tmp2.reshape(1,1);


               // predict digit
               int res = model->predict( row );

               // display
               char prediction[2];
               sprintf( prediction, "%d", res);

               if(DEBUG){
                  LOGI("prediction = %s", prediction);
               }
      
               putText(scaledDown, prediction, center[i], CV_FONT_HERSHEY_COMPLEX, 2, color);

             } // END if

         }// END for


      }// END if(contours.size() > 0)







    /*


       //-- Predict number  
       t = static_cast<double>(getTickCount());
     
        

       t = 1000*(static_cast<double>(getTickCount()) - t)/getTickFrequency();
       if(DEBUG){
          LOGI("Predict time = %0.2f ms.", t);
      }


    */   


     if(pScaledUp == NULL)
       pScaledUp = new Mat(srcGray.size(), srcGray.type());
 
     Mat scaledUp = *pScaledUp;
     pyrUp( scaledDown, scaledUp);


       /// Display to Android
       cvtColor(scaledUp, mbgra, CV_GRAY2BGRA);


      if(DEBUG){
        LOGI("Successfully finished native image processing...");
      }
   
/************************************************************************************************/ 
   
   /// Release Java byte buffer and unlock backing bitmap
   pEnv-> ReleasePrimitiveArrayCritical(pSource,source,0);
   if (AndroidBitmap_unlockPixels(pEnv, pTarget) < 0) abort();
}
