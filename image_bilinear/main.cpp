//˫������ֵ\����������ֵ
//#include <opencv2/opencv.hpp>
//using namespace cv;

//int main() {
//    // ��ȡ����Ҷ�ͼ��
//    cv::Mat inputImage = cv::imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\Mono8_135_Degree_20_28_2.bmp", cv::IMREAD_GRAYSCALE);
//    if (inputImage.empty()) {
//        std::cout << "Could not open or find the image!" << std::endl;
//        return -1;
//    }
//
//    cv::Mat outputImage;
//    resize(inputImage, outputImage, Size(2448, 2048),INTER_CUBIC);
//
//    // ִ��ͼ���ع�
//    //cv::Mat outputImage = reconstructImage(inputImage);
//
//    // ��ʾ��������ͼ��
//    namedWindow("Input Image", WINDOW_NORMAL);
//    namedWindow("Reconstructed Image", WINDOW_NORMAL);
//    cv::imshow("Input Image", inputImage);
//    cv::imshow("Reconstructed Image", outputImage);
//
//    cv::waitKey(0);
//    cv::destroyAllWindows();
//    imwrite("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\12.bmp", outputImage);
//
//    return 0;
//}



//nearest
//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//
//// ͼ���ع�����
//cv::Mat reconstructImage(const cv::Mat& inputImage) {
//    int width = inputImage.cols;
//    int height = inputImage.rows;
//
//    cv::Mat outputImage(height * 2, width * 2, inputImage.type());
//
//    for (int y = 0; y < height; ++y) {
//        for (int x = 0; x < width; ++x) {
//            outputImage.at<uchar>(2 * y, 2 * x) = inputImage.at<uchar>(y, x);
//
//            // O(2p+1, 2q)
//            if (x < width - 1) {
//                outputImage.at<uchar>(2 * y, 2 * x + 1) = inputImage.at<uchar>(y, x);
//            }
//
//            // O(2p, 2q+1)
//            if (y < height - 1) {
//                outputImage.at<uchar>(2 * y + 1, 2 * x) = inputImage.at<uchar>(y, x);
//            }
//
//            // O(2p+1, 2q+1)
//            if (x < width - 1 && y < height - 1) {
//                outputImage.at<uchar>(2 * y + 1, 2 * x + 1) = inputImage.at<uchar>(y, x);
//            }
//        }
//    }
//
//    return outputImage;
//}
//
//int main() {
//    // ��ȡ����Ҷ�ͼ��
//    cv::Mat inputImage = cv::imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\Mono8_135_Degree_20_28_2.bmp", cv::IMREAD_GRAYSCALE);
//    if (inputImage.empty()) {
//        std::cout << "Could not open or find the image!" << std::endl;
//        return -1;
//    }
//
//    // ִ��ͼ���ع�
//    cv::Mat outputImage = reconstructImage(inputImage);
//
//    // ��ʾ��������ͼ��
//    namedWindow("Input Image", WINDOW_NORMAL);
//    namedWindow("Reconstructed Image", WINDOW_NORMAL);
//    cv::imshow("Input Image", inputImage);
//    cv::imshow("Reconstructed Image", outputImage);
//
//    cv::waitKey(0);
//    cv::destroyAllWindows();
//    imwrite("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\8.bmp", outputImage);
//
//    return 0;
//}


//bilinear
//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//
//// ͼ���ع�����
//cv::Mat reconstructImage(const cv::Mat& inputImage) {
//    int width = inputImage.cols;
//    int height = inputImage.rows;
//
//    cv::Mat outputImage(height * 2, width * 2, inputImage.type());
//
//    for (int y = 0; y < height; ++y) {
//        for (int x = 0; x < width; ++x) {
//            outputImage.at<uchar>(2 * y, 2 * x) = inputImage.at<uchar>(y, x);
//
//            // O(2p+1, 2q)
//            if (x < width - 1) {
//                outputImage.at<uchar>(2 * y, 2 * x + 1) = 0.5 * (inputImage.at<uchar>(y, x) + inputImage.at<uchar>(y, x + 1));
//            }
//
//            // O(2p, 2q+1)
//            if (y < height - 1) {
//                outputImage.at<uchar>(2 * y + 1, 2 * x) = 0.5 * (inputImage.at<uchar>(y, x) + inputImage.at<uchar>(y + 1, x));
//            }
//
//            // O(2p+1, 2q+1)
//            if (x < width - 1 && y < height - 1) {
//                outputImage.at<uchar>(2 * y + 1, 2 * x + 1) = 0.25 * (inputImage.at<uchar>(y, x) + inputImage.at<uchar>(y, x + 1) +
//                    inputImage.at<uchar>(y + 1, x) + inputImage.at<uchar>(y + 1, x + 1));
//            }
//        }
//    }
//
//    return outputImage;
//}
//
//int main() {
//    // ��ȡ����Ҷ�ͼ��
//    cv::Mat inputImage = cv::imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\Mono8_135_Degree_20_28_2.bmp", cv::IMREAD_GRAYSCALE);
//    if (inputImage.empty()) {
//        std::cout << "Could not open or find the image!" << std::endl;
//        return -1;
//    }
//
//    // ִ��ͼ���ع�
//    cv::Mat outputImage = reconstructImage(inputImage);
//
//    // ��ʾ��������ͼ��
//    namedWindow("Input Image", WINDOW_NORMAL);
//    namedWindow("Reconstructed Image", WINDOW_NORMAL);
//    cv::imshow("Input Image", inputImage);
//    cv::imshow("Reconstructed Image", outputImage);
//
//    cv::waitKey(0);
//    cv::destroyAllWindows();
//    imwrite("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\135.bmp", outputImage);
//
//    return 0;
//}
//


//#include <opencv2/opencv.hpp>
//#include <opencv2/features2d.hpp>
//
//using namespace cv;
//
//// ͼ���ع�����
//cv::Mat reconstructImage(const cv::Mat& inputImage) {
//    int width = inputImage.cols;
//    int height = inputImage.rows;
//
//    cv::Mat outputImage(height, width, inputImage.type());
//
//    for (int y = 0; y < height/2; ++y) {
//        for (int x = 0; x < width/2; ++x) {
//            // O(2p, 2q) = I(2p, 2q)
//            outputImage.at<uchar>(2*y, 2*x) = inputImage.at<uchar>(2 * y, 2 * x);
//
//            // O(2p+1, 2q)
//            if (2 * y + 1 < height) {
//                outputImage.at<uchar>(2 * y + 1, 2 * x) = 0.5 * inputImage.at<uchar>(2 * y, 2 * x) +
//                    0.5 * inputImage.at<uchar>(std::min(2 * y + 2, height - 1), 2 * x);
//            }
//
//            // O(2p, 2q+1)
//            if (2 * x + 1 < width) {
//                outputImage.at<uchar>(2 * y, 2 * x + 1) = 0.5 * inputImage.at<uchar>(2 * y, 2 * x) +
//                    0.5 * inputImage.at<uchar>(2 * y, std::min(2 * x + 2, width - 1));
//            }
//
//            // O(2p+1, 2q+1)
//            if (2 * y + 1 < height && 2 * x + 1 < width) {
//                outputImage.at<uchar>(2 * y + 1, 2 * x + 1) = 0.25 * inputImage.at<uchar>(2 * y, 2 * x) +
//                    0.25 * inputImage.at<uchar>(std::min(2 * y + 2, height - 1), 2 * x) +
//                    0.25 * inputImage.at<uchar>(2 * y, std::min(2 * x + 2, width - 1)) +
//                    0.25 * inputImage.at<uchar>(std::min(2 * y + 2, height - 1), std::min(2 * x + 2, width - 1));
//            }
//        }
//    }
//
//    return outputImage;
//}
//
//int main() {
//    // ��ȡ����Ҷ�ͼ��
//    cv::Mat inputImage = cv::imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\Mono8_135_Degree_20_28_2.bmp", cv::IMREAD_GRAYSCALE);
//    
//
//
//    // ִ��ͼ���ع�
//    cv::Mat outputImage = reconstructImage(inputImage);
//
//    // ��ʾ��������ͼ��
//    /*namedWindow("Input Image", WINDOW_NORMAL);
//    namedWindow("Reconstructed Image", WINDOW_NORMAL);
//    cv::imshow("Input Image", inputImage);
//    cv::imshow("Reconstructed Image", outputImage);
//    cv::waitKey(0);
//    cv::destroyAllWindows();*/
//
//    //imwrite("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\1351.bmp", outputImage);
//
//    return 0;
//}
