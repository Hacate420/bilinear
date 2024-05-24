//基于边缘的插值
#include <opencv2/opencv.hpp>

using namespace cv;

// 边缘插值函数
Mat edgeAwareInterpolation(const Mat& inputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;

    Mat outputImage(height * 2, width * 2, inputImage.type());

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // 直接复制原始像素值
            outputImage.at<uchar>(2 * y, 2 * x) = inputImage.at<uchar>(y, x);

            // 边缘水平插值
            if (x < width - 1) {
                // 判断边缘
                if (abs(inputImage.at<uchar>(y, x + 1) - inputImage.at<uchar>(y, x)) > 30) {
                    outputImage.at<uchar>(2 * y, 2 * x + 1) = inputImage.at<uchar>(y, x + 1);
                }
                else {
                    // 插值
                    outputImage.at<uchar>(2 * y, 2 * x + 1) = 0.5 * (inputImage.at<uchar>(y, x) + inputImage.at<uchar>(y, x + 1));
                }
            }

            // 边缘垂直插值
            if (y < height - 1) {
                // 判断边缘
                if (abs(inputImage.at<uchar>(y + 1, x) - inputImage.at<uchar>(y, x)) > 30) {
                    outputImage.at<uchar>(2 * y + 1, 2 * x) = inputImage.at<uchar>(y + 1, x);
                }
                else {
                    // 插值
                    outputImage.at<uchar>(2 * y + 1, 2 * x) = 0.5 * (inputImage.at<uchar>(y, x) + inputImage.at<uchar>(y + 1, x));
                }
            }

            // 对角线插值
            if (x < width - 1 && y < height - 1) {
                // 判断边缘
                if (abs(inputImage.at<uchar>(y, x + 1) - inputImage.at<uchar>(y, x)) > 30 ||
                    abs(inputImage.at<uchar>(y + 1, x) - inputImage.at<uchar>(y, x)) > 30) {
                    outputImage.at<uchar>(2 * y + 1, 2 * x + 1) = inputImage.at<uchar>(y + 1, x + 1);
                }
                else {
                    // 插值
                    outputImage.at<uchar>(2 * y + 1, 2 * x + 1) = 0.25 * (inputImage.at<uchar>(y, x) + inputImage.at<uchar>(y, x + 1) +
                        inputImage.at<uchar>(y + 1, x) + inputImage.at<uchar>(y + 1, x + 1));
                }
            }
        }
    }

    return outputImage;
}

//int main() {
//    // 读取输入灰度图像
//    Mat inputImage = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\Mono8_0_Degree_20_28_2.bmp", IMREAD_GRAYSCALE);
//    if (inputImage.empty()) {
//        std::cout << "无法打开或找到图像！" << std::endl;
//        return -1;
//    }
//
//    // 执行图像重构
//    Mat outputImage = edgeAwareInterpolation(inputImage);
//
//    // 显示输入和输出图像
//    namedWindow("Input Image", WINDOW_NORMAL);
//    namedWindow("Reconstructed Image", WINDOW_NORMAL);
//    imshow("Input Image", inputImage);
//    imshow("Reconstructed Image", outputImage);
//
//    waitKey(0);
//    destroyAllWindows();
//    imwrite("F:\\偏振成像\\测试数据\\数据采集\\金属6\\25.bmp", outputImage);
//
//    return 0;
//}





//guass_bilinear（不太行）
//#include <opencv2/opencv.hpp>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//// 计算像素点的权值
//float computeWeight(const Mat& inputImage, int y, int x, float sigma) {
//    Mat patch;
//    getRectSubPix(inputImage, Size(3, 3), Point(x, y), patch); // 提取3x3邻域
//
//    // 计算邻域均值
//    Scalar meanVal = mean(patch);
//
//    // 计算高斯权值
//    float weight = exp(-0.5 * pow((inputImage.at<uchar>(y, x) - meanVal[0]) / sigma, 2));
//
//    return weight;
//}
//
//// 图像重构函数
//Mat reconstructImage(const Mat& inputImage, float sigma) {
//    int width = inputImage.cols;
//    int height = inputImage.rows;
//
//    Mat outputImage(height * 2, width * 2, inputImage.type());
//
//    for (int y = 0; y < height; ++y) {
//        for (int x = 0; x < width; ++x) {
//            outputImage.at<uchar>(2 * y, 2 * x) = inputImage.at<uchar>(y, x);
//
//            // O(2p+1, 2q)
//            if (x < width - 1) {
//                float weight = computeWeight(inputImage, y, x, sigma);
//                outputImage.at<uchar>(2 * y, 2 * x + 1) = weight * 0.5 * (inputImage.at<uchar>(y, x) + inputImage.at<uchar>(y, x + 1));
//            }
//
//            // O(2p, 2q+1)
//            if (y < height - 1) {
//                float weight = computeWeight(inputImage, y, x, sigma);
//                outputImage.at<uchar>(2 * y + 1, 2 * x) = weight * 0.5 * (inputImage.at<uchar>(y, x) + inputImage.at<uchar>(y + 1, x));
//            }
//
//            // O(2p+1, 2q+1)
//            if (x < width - 1 && y < height - 1) {
//                float weight = computeWeight(inputImage, y, x, sigma);
//                outputImage.at<uchar>(2 * y + 1, 2 * x + 1) = weight * 0.25 * (inputImage.at<uchar>(y, x) + inputImage.at<uchar>(y, x + 1) +
//                    inputImage.at<uchar>(y + 1, x) + inputImage.at<uchar>(y + 1, x + 1));
//            }
//        }
//    }
//
//    return outputImage;
//}

//int main() {
//    // 读取输入灰度图像
//    Mat inputImage = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\Mono8_135_Degree_20_28_2.bmp", IMREAD_GRAYSCALE);
//    if (inputImage.empty()) {
//        cout << "无法打开或找到图像！" << endl;
//        return -1;
//    }
//
//    // 执行图像重构，设置高斯滤波器参数
//    float sigma = 1.0;
//    Mat outputImage = reconstructImage(inputImage, sigma);
//
//    // 显示输入和输出图像
//    namedWindow("输入图像", WINDOW_NORMAL);
//    namedWindow("重建图像", WINDOW_NORMAL);
//    imshow("输入图像", inputImage);
//    imshow("重建图像", outputImage);
//    waitKey(0);
//    destroyAllWindows();
//
//    imwrite("F:\\偏振成像\\测试数据\\数据采集\\金属6\\20.bmp", outputImage);
//
//    return 0;
//}



//Soble_Bilinear
//#include <opencv2/opencv.hpp>
//#include "iostream"
//
//using namespace std;
//using namespace cv;
//
//// 计算水平和垂直梯度
//void calculateGradients(const Mat& inputImage, Mat& gradientX, Mat& gradientY) {
//    Sobel(inputImage, gradientX, CV_32F, 1, 0);
//    Sobel(inputImage, gradientY, CV_32F, 0, 1);
//}
//
//// 图像重构函数
//Mat reconstructImage(const Mat& inputImage) {
//    int width = inputImage.cols;
//    int height = inputImage.rows;
//
//    // 计算梯度
//    Mat gradientX, gradientY;
//    calculateGradients(inputImage, gradientX, gradientY);
//
//    Mat outputImage(height * 2, width * 2, inputImage.type());
//
//    for (int y = 0; y < height; ++y) {
//        for (int x = 0; x < width; ++x) {
//            outputImage.at<uchar>(2 * y, 2 * x) = inputImage.at<uchar>(y, x);
//
//            // O(2p+1, 2q)
//            if (x < width - 1) {
//                float weight = 0.5 * (1.0 + abs(gradientX.at<float>(y, x)) / 255.0);
//                outputImage.at<uchar>(2 * y, 2 * x + 1) = weight * inputImage.at<uchar>(y, x) + (1 - weight) * inputImage.at<uchar>(y, x + 1);
//            }
//
//            // O(2p, 2q+1)
//            if (y < height - 1) {
//                float weight = 0.5 * (1.0 + abs(gradientY.at<float>(y, x)) / 255.0);
//                outputImage.at<uchar>(2 * y + 1, 2 * x) = weight * inputImage.at<uchar>(y, x) + (1 - weight) * inputImage.at<uchar>(y + 1, x);
//            }
//
//            // O(2p+1, 2q+1)
//            if (x < width - 1 && y < height - 1) {
//                float weight_x = 0.5 * (1.0 + abs(gradientX.at<float>(y, x)) / 255.0);
//                float weight_y = 0.5 * (1.0 + abs(gradientY.at<float>(y, x)) / 255.0);
//                outputImage.at<uchar>(2 * y + 1, 2 * x + 1) = (weight_x * weight_y) * inputImage.at<uchar>(y, x) +
//                    (1 - weight_x) * weight_y * inputImage.at<uchar>(y, x + 1) +
//                    weight_x * (1 - weight_y) * inputImage.at<uchar>(y + 1, x) +
//                    (1 - weight_x) * (1 - weight_y) * inputImage.at<uchar>(y + 1, x + 1);
//            }
//        }
//    }
//
//    return outputImage;
//}

//int main() {
//    // 读取输入灰度图像
//    Mat inputImage = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\Mono8_135_Degree_20_28_2.bmp", IMREAD_GRAYSCALE);
//    if (inputImage.empty()) {
//        cout << "Could not open or find the image!" << endl;
//        return -1;
//    }
//
//    // 执行图像重构
//    Mat outputImage = reconstructImage(inputImage);
//
//    // 显示输入和输出图像
//    namedWindow("Input Image", WINDOW_NORMAL);
//    namedWindow("Reconstructed Image", WINDOW_NORMAL);
//    imshow("Input Image", inputImage);
//    imshow("Reconstructed Image", outputImage);
//
//    waitKey(0);
//    destroyAllWindows();
//    imwrite("F:\\偏振成像\\测试数据\\数据采集\\金属6\\16.bmp", outputImage);
//
//    return 0;
//}



//#include <opencv2/opencv.hpp>  
//#include <iostream>  
//
//cv::Mat bilinearInterpolation(const cv::Mat& src, int scale) {
//    int width = src.cols * scale;
//    int height = src.rows * scale;
//    cv::Mat dst(height, width, src.type());
//
//    for (int y = 0; y < height; ++y) {
//        float fy = y / (float)scale;
//        int y0 = cv::saturate_cast<int>(std::floor(fy));
//        int y1 = (y0 + 1 < src.rows) ? y0 + 1 : y0;
//        fy -= y0;
//
//        for (int x = 0; x < width; ++x) {
//            float fx = x / (float)scale;
//            int x0 = cv::saturate_cast<int>(std::floor(fx));
//            int x1 = (x0 + 1 < src.cols) ? x0 + 1 : x0;
//            fx -= x0;
//
//            // 获取四个最近邻像素的灰度值  
//            cv::Vec3b tl = src.at<cv::Vec3b>(y0, x0);
//            cv::Vec3b tr = src.at<cv::Vec3b>(y0, x1);
//            cv::Vec3b bl = src.at<cv::Vec3b>(y1, x0);
//            cv::Vec3b br = src.at<cv::Vec3b>(y1, x1);
//
//            // 沿x方向进行线性插值  
//            cv::Vec3b row0 = tl * (1.0f - fx) + tr * fx;
//            cv::Vec3b row1 = bl * (1.0f - fx) + br * fx;
//
//            // 沿y方向进行线性插值  
//            dst.at<cv::Vec3b>(y, x) = row0 * (1.0f - fy) + row1 * fy;
//        }
//    }
//
//    return dst;
//}
//
//int main() {
//    cv::Mat src = cv::imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\Mono8_135_Degree_20_28_2.bmp");
//    if (src.empty()) {
//        std::cerr << "Failed to load image!" << std::endl;
//        return -1;
//    }
//
//    int scale = 2; // 扩大至原来的两倍  
//    cv::Mat dst = bilinearInterpolation(src, scale);
//    cv::namedWindow("Input Image", CV_WINDOW_NORMAL);
//    cv::imshow("Input Image", dst);
//    cv::waitKey(0);
//    cv::imwrite("F:\\偏振成像\\测试数据\\数据采集\\金属6\\4.bmp", dst);
//    //std::cout << "Image resized successfully!" << std::endl;
//
//    return 0;
//}