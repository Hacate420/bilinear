#include <opencv2/opencv.hpp>  
#include <iostream>  

using namespace cv;
using namespace std;

//PSNR，峰值信噪比
double calculatePSNR(Mat& I1, Mat& I2) { //注意，当两幅图像一样时这个函数计算出来的psnr为0 
    Mat s1;
    absdiff(I1, I2, s1);
    s1.convertTo(s1, CV_32F);//转换为32位的float类型，8位不能计算平方  
    s1 = s1.mul(s1);
    Scalar s = sum(s1);  //计算每个通道的和  
    double sse = s.val[0] + s.val[1] + s.val[2];
    if (sse <= 1e-10) // for small values return zero  
        return 0;
    else
    {
        double mse = sse / (double)(I1.channels() * I1.total()); //  sse/(w*h*3)  
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

//SSIM ，结构相似性
double calculateSSIM(Mat& i1, Mat& i2) {
    const double C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);
    i2.convertTo(I2, d);
    Mat I1_2 = I1.mul(I1);
    Mat I2_2 = I2.mul(I2);
    Mat I1_I2 = I1.mul(I2);
    Mat mu1, mu2;
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);
    Mat sigma1_2, sigam2_2, sigam12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigam2_2, Size(11, 11), 1.5);
    sigam2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigam12, Size(11, 11), 1.5);
    sigam12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigam12 + C2;
    t3 = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigam2_2 + C2;
    t1 = t1.mul(t2);

    Mat ssim_map;
    divide(t3, t1, ssim_map);
    Scalar mssim = mean(ssim_map);

    double ssim = (mssim.val[0] + mssim.val[1] + mssim.val[2]) / 3;
    return ssim;
}

/*
单幅图像信息熵计算
定义中，图像的信息熵通常采用灰度图计算
*/
double entropy(Mat& img)
{
    double temp[256] = { 0.0f };
    // 计算每个像素的累积值
    int row = img.rows;
    int col = img.cols;
    for (int r = 0; r < row; r++)
    {
        for (int c = 0; c < col; c++)
        {
            const uchar* i = img.ptr<uchar>(r, c);
            temp[*i] ++;
        }
    }

    // 计算每个像素的概率
    int size = row * col;
    for (int i = 0; i < 256; i++)
    {
        temp[i] = temp[i] / size;
    }

    double result = 0.0f;
    // 计算图像信息熵
    for (int i = 0; i < 256; i++)
    {
        if (temp[i] != 0.0) {
            result += temp[i] * log2(temp[i]);
        }
    }
    return -result;
}

/*
计算平均梯度
梯度的计算应采用灰度图
公式？通用的都是 sqrt( (dx^2 + dy^2) /2  )
笔记中从书《数字图像处理及应用――谢凤英》摘录的公式有待商榷
*/
double meanGradient(Mat& grayImg) {
    if (grayImg.channels() != 1) {
        printf("avgGradient 参数错误，必须输入单通道图！");
        return 0.0;
    }
    // 原灰度图转换成浮点型数据类型
    Mat src;
    grayImg.convertTo(src, CV_64FC1);

    double temp = 0.0f;
    // 由于求一阶差分的边界问题，这里行列都要-1
    int rows = src.rows - 1;
    int cols = src.cols - 1;

    // 根据公式计算平均梯度
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            // 离散的delta就是相邻的离散点的差值
            double dx = src.at<double>(r, c + 1) - src.at<double>(r, c);
            double dy = src.at<double>(r + 1, c) - src.at<double>(r, c);
            double ds = sqrt((dx * dx + dy * dy) / 2);
            temp += ds;
        }
    }
    double imageAVG = temp / (rows * cols);

    return imageAVG;
}

/*计算灰度图的均值和方差*/
void mean_std(const Mat& grayImg, double& mean, double& std) {
    if (grayImg.channels() != 1) {
        printf("mean_std 参数错误，必须输入单通道图！");
        return;
    }
    Mat mat_mean, mat_stddev;
    meanStdDev(grayImg, mat_mean, mat_stddev);
    mean = mat_mean.at<double>(0, 0);
    std = mat_stddev.at<double>(0, 0);
}



//均方差
double getMSE(const Mat& src1, const Mat& src2)
{
    Mat s1;
    absdiff(src1, src2, s1);    // |src1 - src2|
    s1.convertTo(s1, CV_32F);   // 不能在8位矩阵上做平方运算
    s1 = s1.mul(s1);            // |src1 - src2|^2
    Scalar s = sum(s1);         // 叠加每个通道的元素

    double result = 0.0f;
    int ch = s1.channels();
    for (int i = 0; i < ch; i++) {
        // 叠加所有通道
        result += s.val[i];
    }

    if (result <= 1e-10) // 如果值太小就直接等于0
        return 0;
    else
        return result / (ch * s1.total());
}






int main() {
    // 读取原始图像和压缩后的图像
    //Mat originalImage = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\Mono8_Stoke_DoP_20_28_2.bmp", IMREAD_COLOR);
    Mat originalImage = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\原图Dop.bmp", IMREAD_COLOR);
    //Mat originalImage = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\放大Dop.bmp", IMREAD_COLOR);
    resize(originalImage, originalImage, Size(2448, 2048));
    //Mat compressedImage = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\插值Dop.bmp", IMREAD_COLOR);
    //Mat compressedImage = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\放大Dop.bmp", IMREAD_COLOR);
    //Mat compressedImage = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\放大Dop2.bmp", IMREAD_COLOR);
    //Mat compressedImage = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\Soble_双线性Dop.bmp", IMREAD_COLOR);
    //Mat compressedImage = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\边缘插值Dop.bmp", IMREAD_COLOR);
    Mat compressedImage = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\边缘插值Dop2.bmp", IMREAD_COLOR);
   
    Mat Mono8_Dop = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\Mono8_Stoke_DoP_20_28_2.bmp", 0);
    Mat Mono8_Dop_2448;
    resize(Mono8_Dop, Mono8_Dop_2448, Size(2448, 2048));
    Mat YTDop = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\原图Dop.bmp", 0);
    Mat YTDop_2448;
    resize(YTDop, YTDop_2448, Size(2448, 2048));
    Mat CZDop = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\插值Dop.bmp", 0);
    Mat CZDop_2448;
    resize(CZDop, CZDop_2448, Size(2448, 2048));
    Mat FDDop1 = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\放大Dop.bmp", 0);
    Mat FDDop2 = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\放大Dop2.bmp", 0);
    Mat NearDop = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\NearstDop.bmp", 0);
    Mat CubicDop = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\CubicDop.bmp", 0);
    Mat Soble_Dop = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\Soble_双线性Dop.bmp", 0);
    Mat BY_Dop = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\边缘插值Dop2.bmp", 0);
    
    
    if (originalImage.empty() || compressedImage.empty()) {
        cout << "Could not open or find the images!" << endl;
        return -1;
    }

    // 计算PSNR和SSIM
    double psnr = calculatePSNR(originalImage, compressedImage);
    double ssim = calculateSSIM(originalImage, compressedImage);

    //信息熵
    double value1 = entropy(BY_Dop);
    //平均梯度
    double value2 = meanGradient(BY_Dop);
    //均值、方差
    double value3, value4;
    mean_std(BY_Dop, value3, value4);
    //均方差
    double value5 = getMSE(YTDop_2448, BY_Dop);
    // 打印结果
    cout << "PSNR: " << psnr << endl;
    cout << "SSIM: " << ssim << endl;
    cout << "信息熵: " << value1 << endl;
    cout << "平均梯度: " << value2 << endl;
    cout << "均值: " << value3 << endl;
    cout << "方差: " << value4 << endl;
    cout << "均方差: " << value5 << endl;

    return 0;
}