#include <opencv2/opencv.hpp>  
#include <iostream>  

using namespace cv;
using namespace std;

//PSNR����ֵ�����
double calculatePSNR(Mat& I1, Mat& I2) { //ע�⣬������ͼ��һ��ʱ����������������psnrΪ0 
    Mat s1;
    absdiff(I1, I2, s1);
    s1.convertTo(s1, CV_32F);//ת��Ϊ32λ��float���ͣ�8λ���ܼ���ƽ��  
    s1 = s1.mul(s1);
    Scalar s = sum(s1);  //����ÿ��ͨ���ĺ�  
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

//SSIM ���ṹ������
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
����ͼ����Ϣ�ؼ���
�����У�ͼ�����Ϣ��ͨ�����ûҶ�ͼ����
*/
double entropy(Mat& img)
{
    double temp[256] = { 0.0f };
    // ����ÿ�����ص��ۻ�ֵ
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

    // ����ÿ�����صĸ���
    int size = row * col;
    for (int i = 0; i < 256; i++)
    {
        temp[i] = temp[i] / size;
    }

    double result = 0.0f;
    // ����ͼ����Ϣ��
    for (int i = 0; i < 256; i++)
    {
        if (temp[i] != 0.0) {
            result += temp[i] * log2(temp[i]);
        }
    }
    return -result;
}

/*
����ƽ���ݶ�
�ݶȵļ���Ӧ���ûҶ�ͼ
��ʽ��ͨ�õĶ��� sqrt( (dx^2 + dy^2) /2  )
�ʼ��д��顶����ͼ����Ӧ�èD�Dл��Ӣ��ժ¼�Ĺ�ʽ�д���ȶ
*/
double meanGradient(Mat& grayImg) {
    if (grayImg.channels() != 1) {
        printf("avgGradient �������󣬱������뵥ͨ��ͼ��");
        return 0.0;
    }
    // ԭ�Ҷ�ͼת���ɸ�������������
    Mat src;
    grayImg.convertTo(src, CV_64FC1);

    double temp = 0.0f;
    // ������һ�ײ�ֵı߽����⣬�������ж�Ҫ-1
    int rows = src.rows - 1;
    int cols = src.cols - 1;

    // ���ݹ�ʽ����ƽ���ݶ�
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            // ��ɢ��delta�������ڵ���ɢ��Ĳ�ֵ
            double dx = src.at<double>(r, c + 1) - src.at<double>(r, c);
            double dy = src.at<double>(r + 1, c) - src.at<double>(r, c);
            double ds = sqrt((dx * dx + dy * dy) / 2);
            temp += ds;
        }
    }
    double imageAVG = temp / (rows * cols);

    return imageAVG;
}

/*����Ҷ�ͼ�ľ�ֵ�ͷ���*/
void mean_std(const Mat& grayImg, double& mean, double& std) {
    if (grayImg.channels() != 1) {
        printf("mean_std �������󣬱������뵥ͨ��ͼ��");
        return;
    }
    Mat mat_mean, mat_stddev;
    meanStdDev(grayImg, mat_mean, mat_stddev);
    mean = mat_mean.at<double>(0, 0);
    std = mat_stddev.at<double>(0, 0);
}



//������
double getMSE(const Mat& src1, const Mat& src2)
{
    Mat s1;
    absdiff(src1, src2, s1);    // |src1 - src2|
    s1.convertTo(s1, CV_32F);   // ������8λ��������ƽ������
    s1 = s1.mul(s1);            // |src1 - src2|^2
    Scalar s = sum(s1);         // ����ÿ��ͨ����Ԫ��

    double result = 0.0f;
    int ch = s1.channels();
    for (int i = 0; i < ch; i++) {
        // ��������ͨ��
        result += s.val[i];
    }

    if (result <= 1e-10) // ���ֵ̫С��ֱ�ӵ���0
        return 0;
    else
        return result / (ch * s1.total());
}






int main() {
    // ��ȡԭʼͼ���ѹ�����ͼ��
    //Mat originalImage = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\Mono8_Stoke_DoP_20_28_2.bmp", IMREAD_COLOR);
    Mat originalImage = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\ԭͼDop.bmp", IMREAD_COLOR);
    //Mat originalImage = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\�Ŵ�Dop.bmp", IMREAD_COLOR);
    resize(originalImage, originalImage, Size(2448, 2048));
    //Mat compressedImage = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\��ֵDop.bmp", IMREAD_COLOR);
    //Mat compressedImage = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\�Ŵ�Dop.bmp", IMREAD_COLOR);
    //Mat compressedImage = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\�Ŵ�Dop2.bmp", IMREAD_COLOR);
    //Mat compressedImage = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\Soble_˫����Dop.bmp", IMREAD_COLOR);
    //Mat compressedImage = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\��Ե��ֵDop.bmp", IMREAD_COLOR);
    Mat compressedImage = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\��Ե��ֵDop2.bmp", IMREAD_COLOR);
   
    Mat Mono8_Dop = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\Mono8_Stoke_DoP_20_28_2.bmp", 0);
    Mat Mono8_Dop_2448;
    resize(Mono8_Dop, Mono8_Dop_2448, Size(2448, 2048));
    Mat YTDop = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\ԭͼDop.bmp", 0);
    Mat YTDop_2448;
    resize(YTDop, YTDop_2448, Size(2448, 2048));
    Mat CZDop = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\��ֵDop.bmp", 0);
    Mat CZDop_2448;
    resize(CZDop, CZDop_2448, Size(2448, 2048));
    Mat FDDop1 = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\�Ŵ�Dop.bmp", 0);
    Mat FDDop2 = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\�Ŵ�Dop2.bmp", 0);
    Mat NearDop = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\NearstDop.bmp", 0);
    Mat CubicDop = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\CubicDop.bmp", 0);
    Mat Soble_Dop = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\Soble_˫����Dop.bmp", 0);
    Mat BY_Dop = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\��Ե��ֵDop2.bmp", 0);
    
    
    if (originalImage.empty() || compressedImage.empty()) {
        cout << "Could not open or find the images!" << endl;
        return -1;
    }

    // ����PSNR��SSIM
    double psnr = calculatePSNR(originalImage, compressedImage);
    double ssim = calculateSSIM(originalImage, compressedImage);

    //��Ϣ��
    double value1 = entropy(BY_Dop);
    //ƽ���ݶ�
    double value2 = meanGradient(BY_Dop);
    //��ֵ������
    double value3, value4;
    mean_std(BY_Dop, value3, value4);
    //������
    double value5 = getMSE(YTDop_2448, BY_Dop);
    // ��ӡ���
    cout << "PSNR: " << psnr << endl;
    cout << "SSIM: " << ssim << endl;
    cout << "��Ϣ��: " << value1 << endl;
    cout << "ƽ���ݶ�: " << value2 << endl;
    cout << "��ֵ: " << value3 << endl;
    cout << "����: " << value4 << endl;
    cout << "������: " << value5 << endl;

    return 0;
}