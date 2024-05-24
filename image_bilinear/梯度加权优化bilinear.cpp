//���ڱ�Ե�Ĳ�ֵ
#include <opencv2/opencv.hpp>

using namespace cv;

// ��Ե��ֵ����
Mat edgeAwareInterpolation(const Mat& inputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;

    Mat outputImage(height * 2, width * 2, inputImage.type());

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // ֱ�Ӹ���ԭʼ����ֵ
            outputImage.at<uchar>(2 * y, 2 * x) = inputImage.at<uchar>(y, x);

            // ��Եˮƽ��ֵ
            if (x < width - 1) {
                // �жϱ�Ե
                if (abs(inputImage.at<uchar>(y, x + 1) - inputImage.at<uchar>(y, x)) > 30) {
                    outputImage.at<uchar>(2 * y, 2 * x + 1) = inputImage.at<uchar>(y, x + 1);
                }
                else {
                    // ��ֵ
                    outputImage.at<uchar>(2 * y, 2 * x + 1) = 0.5 * (inputImage.at<uchar>(y, x) + inputImage.at<uchar>(y, x + 1));
                }
            }

            // ��Ե��ֱ��ֵ
            if (y < height - 1) {
                // �жϱ�Ե
                if (abs(inputImage.at<uchar>(y + 1, x) - inputImage.at<uchar>(y, x)) > 30) {
                    outputImage.at<uchar>(2 * y + 1, 2 * x) = inputImage.at<uchar>(y + 1, x);
                }
                else {
                    // ��ֵ
                    outputImage.at<uchar>(2 * y + 1, 2 * x) = 0.5 * (inputImage.at<uchar>(y, x) + inputImage.at<uchar>(y + 1, x));
                }
            }

            // �Խ��߲�ֵ
            if (x < width - 1 && y < height - 1) {
                // �жϱ�Ե
                if (abs(inputImage.at<uchar>(y, x + 1) - inputImage.at<uchar>(y, x)) > 30 ||
                    abs(inputImage.at<uchar>(y + 1, x) - inputImage.at<uchar>(y, x)) > 30) {
                    outputImage.at<uchar>(2 * y + 1, 2 * x + 1) = inputImage.at<uchar>(y + 1, x + 1);
                }
                else {
                    // ��ֵ
                    outputImage.at<uchar>(2 * y + 1, 2 * x + 1) = 0.25 * (inputImage.at<uchar>(y, x) + inputImage.at<uchar>(y, x + 1) +
                        inputImage.at<uchar>(y + 1, x) + inputImage.at<uchar>(y + 1, x + 1));
                }
            }
        }
    }

    return outputImage;
}

//int main() {
//    // ��ȡ����Ҷ�ͼ��
//    Mat inputImage = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\Mono8_0_Degree_20_28_2.bmp", IMREAD_GRAYSCALE);
//    if (inputImage.empty()) {
//        std::cout << "�޷��򿪻��ҵ�ͼ��" << std::endl;
//        return -1;
//    }
//
//    // ִ��ͼ���ع�
//    Mat outputImage = edgeAwareInterpolation(inputImage);
//
//    // ��ʾ��������ͼ��
//    namedWindow("Input Image", WINDOW_NORMAL);
//    namedWindow("Reconstructed Image", WINDOW_NORMAL);
//    imshow("Input Image", inputImage);
//    imshow("Reconstructed Image", outputImage);
//
//    waitKey(0);
//    destroyAllWindows();
//    imwrite("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\25.bmp", outputImage);
//
//    return 0;
//}





//guass_bilinear����̫�У�
//#include <opencv2/opencv.hpp>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//// �������ص��Ȩֵ
//float computeWeight(const Mat& inputImage, int y, int x, float sigma) {
//    Mat patch;
//    getRectSubPix(inputImage, Size(3, 3), Point(x, y), patch); // ��ȡ3x3����
//
//    // ���������ֵ
//    Scalar meanVal = mean(patch);
//
//    // �����˹Ȩֵ
//    float weight = exp(-0.5 * pow((inputImage.at<uchar>(y, x) - meanVal[0]) / sigma, 2));
//
//    return weight;
//}
//
//// ͼ���ع�����
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
//    // ��ȡ����Ҷ�ͼ��
//    Mat inputImage = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\Mono8_135_Degree_20_28_2.bmp", IMREAD_GRAYSCALE);
//    if (inputImage.empty()) {
//        cout << "�޷��򿪻��ҵ�ͼ��" << endl;
//        return -1;
//    }
//
//    // ִ��ͼ���ع������ø�˹�˲�������
//    float sigma = 1.0;
//    Mat outputImage = reconstructImage(inputImage, sigma);
//
//    // ��ʾ��������ͼ��
//    namedWindow("����ͼ��", WINDOW_NORMAL);
//    namedWindow("�ؽ�ͼ��", WINDOW_NORMAL);
//    imshow("����ͼ��", inputImage);
//    imshow("�ؽ�ͼ��", outputImage);
//    waitKey(0);
//    destroyAllWindows();
//
//    imwrite("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\20.bmp", outputImage);
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
//// ����ˮƽ�ʹ�ֱ�ݶ�
//void calculateGradients(const Mat& inputImage, Mat& gradientX, Mat& gradientY) {
//    Sobel(inputImage, gradientX, CV_32F, 1, 0);
//    Sobel(inputImage, gradientY, CV_32F, 0, 1);
//}
//
//// ͼ���ع�����
//Mat reconstructImage(const Mat& inputImage) {
//    int width = inputImage.cols;
//    int height = inputImage.rows;
//
//    // �����ݶ�
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
//    // ��ȡ����Ҷ�ͼ��
//    Mat inputImage = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\Mono8_135_Degree_20_28_2.bmp", IMREAD_GRAYSCALE);
//    if (inputImage.empty()) {
//        cout << "Could not open or find the image!" << endl;
//        return -1;
//    }
//
//    // ִ��ͼ���ع�
//    Mat outputImage = reconstructImage(inputImage);
//
//    // ��ʾ��������ͼ��
//    namedWindow("Input Image", WINDOW_NORMAL);
//    namedWindow("Reconstructed Image", WINDOW_NORMAL);
//    imshow("Input Image", inputImage);
//    imshow("Reconstructed Image", outputImage);
//
//    waitKey(0);
//    destroyAllWindows();
//    imwrite("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\16.bmp", outputImage);
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
//            // ��ȡ�ĸ���������صĻҶ�ֵ  
//            cv::Vec3b tl = src.at<cv::Vec3b>(y0, x0);
//            cv::Vec3b tr = src.at<cv::Vec3b>(y0, x1);
//            cv::Vec3b bl = src.at<cv::Vec3b>(y1, x0);
//            cv::Vec3b br = src.at<cv::Vec3b>(y1, x1);
//
//            // ��x����������Բ�ֵ  
//            cv::Vec3b row0 = tl * (1.0f - fx) + tr * fx;
//            cv::Vec3b row1 = bl * (1.0f - fx) + br * fx;
//
//            // ��y����������Բ�ֵ  
//            dst.at<cv::Vec3b>(y, x) = row0 * (1.0f - fy) + row1 * fy;
//        }
//    }
//
//    return dst;
//}
//
//int main() {
//    cv::Mat src = cv::imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\Mono8_135_Degree_20_28_2.bmp");
//    if (src.empty()) {
//        std::cerr << "Failed to load image!" << std::endl;
//        return -1;
//    }
//
//    int scale = 2; // ������ԭ��������  
//    cv::Mat dst = bilinearInterpolation(src, scale);
//    cv::namedWindow("Input Image", CV_WINDOW_NORMAL);
//    cv::imshow("Input Image", dst);
//    cv::waitKey(0);
//    cv::imwrite("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\4.bmp", dst);
//    //std::cout << "Image resized successfully!" << std::endl;
//
//    return 0;
//}