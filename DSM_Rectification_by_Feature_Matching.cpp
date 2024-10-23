// DSM_Rectification_by_Feature_Matching.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#define RELEASE(x)	if(x!=NULL) {delete []x; x = NULL;}
#define _CRT_SECURE_NO_WARNINGS
#define RANSAC_THRESHOLD 2.1
#define WEIGHT_ITERATION_SIGMA_BEF 0.25
#define ABNORMAL_VALUE_THRESHOLD 6
#define RANSAC_RATIO 0.8
#define K_0 2.5
#define K_1 3.5

#include<stdlib.h>
#include<io.h>
#include<iomanip>
#include<string.h>
#include<vector>
#include<stdio.h>
#include <iostream>
#include "gdal_priv.h"
#include <gdal_alg_priv.h>
#include <gdal.h>
#include <memory>
#include <direct.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cassert>  // 包含断言库
#include <random>
#include <fstream>  // 引入文件流库
#include <omp.h>
#include <set>
#include <cstdlib>
#include <ctime>
#include "nanoflann.hpp"


//// 函数生成一个在[0, max_range)范围内的随机数
//int generateRandomNumber(int max_range) {
//    std::random_device rd;  // 获取随机数种子
//    std::mt19937 gen(rd()); // 初始化Mersenne Twister引擎
//    std::uniform_int_distribution<> distrib(0, max_range - 1);
//
//    return distrib(gen); // 返回生成的随机数
//}

int generateRandomNumber(int max_range) // 固定随机种子，结果可复现
{
    static std::mt19937 gen(42); // 使用固定种子初始化Mersenne Twister引擎
    std::uniform_int_distribution<> distrib(0, max_range - 1);

    return distrib(gen); // 返回生成的随机数
}

using namespace std;
using namespace cv;
using namespace nanoflann;

void Image_Chunking(string Input_Dir, string Output_Dir, int Cut_Num, string Image_Type);//Cut_Num需要为平方数，如4，9，16
void Image_Chunking_Process(string All_Area_Dir, int Cut_Num, string exe_path, bool is_Feature_Matching, bool is_Geometric_Correction, bool is_Precision_Evaluation, bool is_Building_Mask, string Mask_Type, bool is_Elevation_Weight);
void Image_Chunking_Process_2017(string All_Area_Dir, int Cut_Num, string exe_path, bool is_Feature_Matching, bool is_Geometric_Correction, bool is_Precision_Evaluation);
void Image_Chunking_Process_2024(string All_Area_Dir, int Cut_Num, string exe_path, bool is_Feature_Matching, bool is_Geometric_Correction, bool is_Precision_Evaluation);
void Image_Chunking_Process_Para_Adjust(string All_Area_Dir, int Cut_Num, string exe_path, bool is_Feature_Matching, bool is_Geometric_Correction, bool is_Precision_Evaluation, bool is_Building_Mask, string Mask_Type, bool is_Elevation_Weight, bool is_RANSAC_Iter, bool is_Basemap_Num);

void GetAllFiles_tif(string path, vector<string>& files);
void GetFolder(string path, vector<string>&folder);
void GetAllFiles_tfw(string path, vector<string>&files);
void GetAllFiles_txt2GeoInfo_BM_and_DSM(string dir_path, string BM_type, vector <vector <Point3d>>&BM_All_Points_input, vector <vector <Point3d>>&DSM_All_Points_input);

void RANSAC_Point(vector <vector<Point3d>> &BM_pt, vector <vector<Point3d>> &DSM_pt, int Init_Iter_Time, double Threshold, double confidence, double Inlier_Ratio,  Point3d &Model_Translation_Parameter_RANSAC, string BM_name, string dirpath);
void RANSAC_Point_Semantic_Constraint(vector <vector<Point3d>>& BM_pt, vector <vector<Point3d>>& DSM_pt, vector <vector<float>> Building_Prob_Weight, int Init_Iter_Time, double Threshold, double confidence, double Inlier_Ratio, Point3d& Model_Translation_Parameter_RANSAC, string BM_name, string dirpath,
                                     string Mask_Path, string Mask_Type, string BM_Type, int Cut_Num);
void Building_Probabilty_Weighting(vector <vector<Point3d>>& BM_pt, vector <vector<Point3d>>& DSM_pt, vector <vector<float>>& Building_Prob_Weight, string Mask_Path, string Mask_Type, string BM_Type, int Cut_Num);
void weightedRandomSample(const std::vector<std::vector<float>>& weights, vector<int>& random_1, vector<int>& random_2, int group_num);

void Weight_Iteration_Point(vector <vector<Point3d>> BM_pt, vector <vector<Point3d>> DSM_pt, double Init_Para_x, double Init_Para_y, double Init_Para_z, double sigma_bef, Point3d &Model_Translation_Parameter_WEIGHT_ITER);

void DOM_Rectified(const char* pszSrcFile, const char* pszDstFile, const char* pszFormat, double x, double y);
void DSM_Rectified(const char* pszSrcFile, const char* pszDstFile, const char* pszFormat, double x, double y, double z);

void Precision_Evaluation(const char* TruthFile, const char* TestFile, string DstDir, string Area, bool is_Write_Heatmap);

void Building_Mask_Elimination_Cross_Error(vector <vector<Point3d>> &BM_pt, vector <vector<Point3d>> &DSM_pt, string Mask_Path, string Mask_Type, string BM_Type, int Cut_Num);

void Weight_Iteration_Point_Elevation(vector <vector<Point3d>> BM_pt, vector <vector<Point3d>> DSM_pt, double Init_Para_x, double Init_Para_y, double Init_Para_z, double sigma_bef, Point3d & Model_Translation_Parameter_WEIGHT_ITER, vector<vector<double>> final_confidenceMatrix, string dir_path);

void Elevation_Weight_Calculation(vector<vector<Point3d>>&BM_pt, vector<vector<Point3d>>&DSM_pt, string Mask_Path, string BM_Type, string dir_path, vector<vector<double>>&Elevation_Weight);
double CalculateConfidenceForCell(GDALDataset * poMaskDS, double cellGeoX, double cellGeoY, double cellSizeX, double cellSizeY);
int GetMaskValue(GDALDataset * poMaskDS, double geoX, double geoY);

void GetAllFiles_txt2GeoInfo_BM_and_DSM_Para_Adjust(string dir_path, string BM_type, vector <vector <Point3d>>& BM_All_Points_input, vector <vector <Point3d>>& DSM_All_Points_input, double error_x, double error_y, double error_z);
void GetAllFiles_txt2GeoInfo_BM_and_DSM_Para_Adjust_Basemap_Num(string dir_path, string BM_type, vector <vector <Point3d>>& BM_All_Points_input, vector <vector <Point3d>>& DSM_All_Points_input, double error_x, double error_y, double error_z, set<vector<int>> basemap_group, int basemap_random_times_i);
vector<int> getVectorFromSet(const set<vector<int>>& basemap_group, int n);
void Elevation_Weight_Calculation_Basemap_Num(vector<vector<Point3d>>& BM_pt, vector<vector<Point3d>>& DSM_pt, string Mask_Path, string BM_Type, string dir_path, vector<vector<double>>& Elevation_Weight, set<vector<int>> basemap_group, int basemap_random_times_i);

void Basemap_Cut(string Input_Basemap_Dir, string Output_Dir);
void ImageCut(const char* pszSrcFile, const char* pszDstFile, int iStartX, int iStartY, int iSizeX, int iSizeY, const char* pszFormat);

bool hasNaN(const Point3d& point);
void removeNaNPoints(std::vector<std::vector<Point3d>>& points1, std::vector<std::vector<Point3d>>& points2);

void loadPointsBinary(const std::string& filename, std::vector<std::vector<Point3d>>& points);
void savePointsBinary(const std::string& filename, const std::vector<std::vector<Point3d>>& points);

double calculateRMSE(const std::vector<Point3d>& truth, const std::vector<Point3d>& data);
double calculateAverageEuclideanDistance_XY(const std::vector<Point3d>& truth, const std::vector<Point3d>& data);
double calculateRMSE_Z(const std::vector<Point3d>& truth, const std::vector<Point3d>& data);
void loadPoints(const std::string& filename, std::vector<Point3d>& points);
void applyCorrection(std::vector<Point3d>& data, const Point3d& correction);
void calculateGeoPositioningAccuracy(const std::string& method, const Point3d& correction, const std::string& truthPath, const std::string& dataPath);


int main(int argc, char** argv)
{
    if (argc < 9 && argc != 6)
    {
        printf("软件名称：联合多时相多源底图的卫星地理产品几何校正软件\n");
        // 匹配时影像为黑色，没有匹配点，则大概率是因为影像的位深度不是8bit
        printf("本计算机软件实现基于多源多时相底图的大范围地理产品（DSM、DOM）的几何校正，支持Google Earth、ArcGIS、SRTM等数据源底图（8bit、WGS84 UTM坐标系、大地高），支持输入该区域建筑物掩膜达到更佳的几何校正精度（uint8的tif），采用分块整体平差方法实现\n");
        printf("该软件仅用于学术科研用途，已获软件著作权，登记号：2023SR0580603，如在使用中遇到问题，请邮件联系luoqy26@mail2.sysu.edu.cn\n");
        printf("Keywords:\n1. All;\n2. Feature_Matching;\n3. Geometric_Correction;\n4. Precision_Evaluation.\n");
        printf("若使用多时相底图控制点优化算法，请依次输入1.Keyword;2.数据所在文件夹;3.match_pairs.exe所在文件夹;4.分块数量(平方数);5、是否已生成该区域的建筑物掩膜(1/ 0);6、建筑物掩膜类型(DOM/ BASEMAP);7、是否使用底图建筑物掩膜来进行高程置信度赋权(1/ 0);8、是否进行调参(1/ 0);9、调参类型(RANSAC_Iter/ Basemap_Num，若不需要调参则不需要输入该参数)。请注意数据需要存放于match_pairs.exe同级的assets文件夹\n");
        printf("若使用多时相底图控制点平均算法，请依次输入1.Keyword;2.数据所在文件夹;3.match_pairs.exe所在文件夹;4.分块数量(平方数);5.方法名称（2017/ 2024）\n");
        return -1;
        // 示例：Geometric_Correction E:\Geometric_Correction_exe\match_pairs\assets\ZHUHAI_AREA E:\Geometric_Correction_exe\match_pairs 9 1 BASEMAP 1 0
    }

    if (argc > 9  && argc != 6)
    {
        if (std::atoi(argv[8]) == 0)
        {
            printf("输入参数不符合要求\n");
            return -1;
        }

    }  

    if (argc < 10 && argc != 6)
    {
        if (std::atoi(argv[8]) == 1)
        {
            printf("输入参数不符合要求\n");
            return -1;
        }
    }

    bool is_Feature_Matching = false;
    bool is_Geometric_Correction = false;
    bool is_Precision_Evaluation = false;
    bool is_2017 = false;
    bool is_2024 = false;

    if (argc == 9 && std::atoi(argv[8]) == 0)
    {
        if (string(argv[1]) == "All")
        {
            is_Feature_Matching = true;
            is_Geometric_Correction = true;
            is_Precision_Evaluation = true;
        }
        else if (string(argv[1]) == "Feature_Matching")
        {
            is_Feature_Matching = true;
            is_Geometric_Correction = false;
            is_Precision_Evaluation = false;
        }
        else if (string(argv[1]) == "Geometric_Correction")
        {
            is_Feature_Matching = false;
            is_Geometric_Correction = true;
            is_Precision_Evaluation = false;
        }
        else if (string(argv[1]) == "Precision_Evaluation")
        {
            is_Feature_Matching = false;
            is_Geometric_Correction = false;
            is_Precision_Evaluation = true;
        }
        else
        {
            printf("Wrong Keyword!\n");
            return -1;
        }
    }

    bool is_RANSAC_Iter = false;
    bool is_Basemap_Num = false;
    if (argc == 10 && std::atoi(argv[8]) == 1)
    {
        is_Feature_Matching = false;
        is_Geometric_Correction = true;
        is_Precision_Evaluation = false;
        if (string(argv[9]) == "RANSAC_Iter")
        {
            is_RANSAC_Iter = true;
        }
        else if (string(argv[9]) == "Basemap_Num")
        {
            is_Basemap_Num = true;
        }
        else
        {
            printf("Wrong Keyword!\n");
            return -1;
        }
    }


    int Cut_Num1 = atoi(argv[4]);

    if (argc == 6)
    {
        if (std::atoi(argv[5]) == 2017)
        {
            is_2017 = true;
        }
        if (std::atoi(argv[5]) == 2024)
        {
            is_2024 = true;
        }
        if (is_2017 || is_2024)
        {
            if (string(argv[1]) == "All")
            {
                is_Feature_Matching = true;
                is_Geometric_Correction = true;
                is_Precision_Evaluation = true;
            }
            else if (string(argv[1]) == "Feature_Matching")
            {
                is_Feature_Matching = true;
                is_Geometric_Correction = false;
                is_Precision_Evaluation = false;
            }
            else if (string(argv[1]) == "Geometric_Correction")
            {
                is_Feature_Matching = false;
                is_Geometric_Correction = true;
                is_Precision_Evaluation = false;
            }
            else if (string(argv[1]) == "Precision_Evaluation")
            {
                is_Feature_Matching = false;
                is_Geometric_Correction = false;
                is_Precision_Evaluation = true;
            }
            else
            {
                printf("Wrong Keyword!\n");
                return -1;
            }
        }

        if (is_2017)
        {
            Image_Chunking_Process_2017(string(argv[2]), Cut_Num1, string(argv[3]), is_Feature_Matching, is_Geometric_Correction, is_Precision_Evaluation);
            return 2017;
        }

        if (is_2024)
        {
            Image_Chunking_Process_2024(string(argv[2]), Cut_Num1, string(argv[3]), is_Feature_Matching, is_Geometric_Correction, is_Precision_Evaluation);
            return 2024;
        }
    }

    bool is_Building_Mask = false;
    if (std::atoi(argv[5]) == 1)
    {
        is_Building_Mask = true;
    }

    bool is_Elevation_Optimized = false;
    if (std::atoi(argv[7]) == 1)
    {
        is_Elevation_Optimized = true;
    }

    if (std::atoi(argv[8]) == 0)
    {
        Image_Chunking_Process(string(argv[2]), Cut_Num1, string(argv[3]), is_Feature_Matching, is_Geometric_Correction, is_Precision_Evaluation, is_Building_Mask, string(argv[6]), is_Elevation_Optimized);
    }
    else
    {
        Image_Chunking_Process_Para_Adjust(string(argv[2]), Cut_Num1, string(argv[3]), is_Feature_Matching, is_Geometric_Correction, is_Precision_Evaluation, is_Building_Mask, string(argv[6]), is_Elevation_Optimized, is_RANSAC_Iter, is_Basemap_Num);
    }
    ////裁切OMA的影像
    //Basemap_Cut("D:\\.match_pairs\\assets\\Omaha_GE\\BASEMAP", "D:\\.match_pairs\\assets\\Omaha_All_Areas_GE");


}

void ImageCut(const char* pszSrcFile, const char* pszDstFile, int iStartX, int iStartY, int iSizeX, int iSizeY, const char* pszFormat)
{
    GDALAllRegister();

    // 检查输出文件是否存在
    std::ifstream file(pszDstFile);
    if (file.good()) {
        // 文件已存在，关闭文件并退出函数
        file.close();
        std::cout << "File " << pszDstFile << " already exists. Skipping writing." << std::endl;
        return;
    }

    GDALDataset* pSrcDS = (GDALDataset*)GDALOpen(pszSrcFile, GA_ReadOnly);
    GDALDataType eDT = pSrcDS->GetRasterBand(1)->GetRasterDataType();

    int iBandCount = pSrcDS->GetRasterCount();

    //根据裁切范围确定裁切后的图像宽高
    int iDstWidth = iSizeX;
    int iDstHeight = iSizeY;

    double adfGeoTransform[6] = { 0 };
    pSrcDS->GetGeoTransform(adfGeoTransform);

    //计算裁切后的图像的左上角坐标
    adfGeoTransform[0] = adfGeoTransform[0] + iStartX * adfGeoTransform[1] + iStartY * adfGeoTransform[2];
    adfGeoTransform[3] = adfGeoTransform[3] + iStartX * adfGeoTransform[4] + iStartY * adfGeoTransform[5];

    //创建输出文件并设置空间参考和坐标信息
    GDALDriver* poDriver = (GDALDriver*)GDALGetDriverByName(pszFormat);
    GDALDataset* pDstDS = poDriver->Create(pszDstFile, iDstWidth, iDstHeight, iBandCount, eDT, NULL);
    pDstDS->SetGeoTransform(adfGeoTransform);
    //pDstDS->SetProjection(pSrcDS->GetProjectionRef());

    int* pBandMap = new int[iBandCount];
    for (int i = 0; i < iBandCount; i++)
    {
        pBandMap[i] = i + 1;
    }

    if (eDT == GDT_Byte)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        unsigned int* pDataBuff = new unsigned int[iDstWidth * iDstHeight * iBandCount];
        pSrcDS->RasterIO(GF_Read, iStartX, iStartY, iSizeX, iSizeY, pDataBuff, iSizeX, iSizeY, eDT, iBandCount, pBandMap, 0, 0, 0);
        pDstDS->RasterIO(GF_Write, 0, 0, iSizeX, iSizeY, pDataBuff, iSizeX, iSizeY, eDT, iBandCount, pBandMap, 0, 0, 0);

        RELEASE(pDataBuff);
    }
    else if (eDT == GDT_UInt16)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        unsigned short* pDataBuff = new unsigned short[iDstWidth * iDstHeight * iBandCount];
        pSrcDS->RasterIO(GF_Read, iStartX, iStartY, iSizeX, iSizeY, pDataBuff, iSizeX, iSizeY, eDT, iBandCount, pBandMap, 0, 0, 0);
        pDstDS->RasterIO(GF_Write, 0, 0, iSizeX, iSizeY, pDataBuff, iSizeX, iSizeY, eDT, iBandCount, pBandMap, 0, 0, 0);

        RELEASE(pDataBuff);
    }
    else if (eDT == GDT_Float32)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        float* pDataBuff = new float[iDstWidth * iDstHeight * iBandCount];
        pSrcDS->RasterIO(GF_Read, iStartX, iStartY, iSizeX, iSizeY, pDataBuff, iSizeX, iSizeY, eDT, iBandCount, pBandMap, 0, 0, 0);
        pDstDS->RasterIO(GF_Write, 0, 0, iSizeX, iSizeY, pDataBuff, iSizeX, iSizeY, eDT, iBandCount, pBandMap, 0, 0, 0);

        RELEASE(pDataBuff);
    }

    RELEASE(pBandMap);

    GDALClose((GDALDatasetH)pSrcDS);
    GDALClose((GDALDatasetH)pDstDS);
    GDALDestroyDriverManager();
    return;
}


void Image_Chunking(string Input_Dir, string Output_Dir, int Cut_Num, string Image_Type)//Cut_Num需要为平方数，如4，9，16
{
    if (Image_Type == "SRTM")
    {
        vector <string> files;
        GetAllFiles_tif(Input_Dir, files);
        for (int i = 0; i < Cut_Num; i++)
        {
            GDALAllRegister();//注册所有的驱动
            CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名

            //1、加载tif数据
            string file_path_name = Input_Dir + "\\" + files[0];

            GDALDataset* poDataset = (GDALDataset*)GDALOpen(file_path_name.c_str(), GA_ReadOnly);//GA_Update和GA_ReadOnly两种模式
            if (poDataset == NULL)
                std::cout << "指定的文件不能打开!" << std::endl;

            //获取图像的尺寸
            int nImgSizeX = poDataset->GetRasterXSize();
            int nImgSizeY = poDataset->GetRasterYSize();
            ImageCut((Input_Dir + "\\" + files[0]).c_str(), (Output_Dir + "\\" + to_string(i) + "\\" + "BASEMAP" + "\\" + Image_Type + "\\" + files[0]).c_str(), 0, 0, nImgSizeX, nImgSizeY, "GTiff");
        }
    }
    if (Image_Type == "ARC_BING_MB" || Image_Type == "GE" )
    {
        vector <string> files;
        GetAllFiles_tif(Input_Dir, files);
        for (int i = 0; i < files.size(); i++)
        {
            GDALAllRegister();//注册所有的驱动
            CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名

            //1、加载tif数据
            string file_path_name = Input_Dir + "\\" + files[i];

            GDALDataset* poDataset = (GDALDataset*)GDALOpen(file_path_name.c_str(), GA_ReadOnly);//GA_Update和GA_ReadOnly两种模式
            if (poDataset == NULL)
                std::cout << "指定的文件不能打开!" << std::endl;

            //获取图像的尺寸
            int nImgSizeX = poDataset->GetRasterXSize();
            int nImgSizeY = poDataset->GetRasterYSize();

            int dx = nImgSizeX / sqrt(Cut_Num);
            int dy = nImgSizeY / sqrt(Cut_Num);

            int Cut_Num_Row_Col = sqrt(Cut_Num);
            for (int ii = 0; ii < Cut_Num_Row_Col; ii++)//按行排序
            {
                for (int jj = 0; jj < Cut_Num_Row_Col; jj++)
                {
                    if (jj == Cut_Num_Row_Col - 1 && ii == Cut_Num_Row_Col - 1)
                    {
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "ARC_BING_MB").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "ARC_BING_MB").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "GE").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "GE").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "SRTM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "SRTM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "ARC_BING_MB").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "ARC_BING_MB").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "GE").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "GE").c_str());
                        }
                        ImageCut((Input_Dir + "\\" + files[i]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + Image_Type + "\\" + files[i]).c_str(), jj * dx, ii * dy, (nImgSizeX - (Cut_Num_Row_Col - 1) * dx), (nImgSizeY - (Cut_Num_Row_Col - 1) * dy), "GTiff");
                    }
                    else if (jj == Cut_Num_Row_Col - 1)
                    {
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "ARC_BING_MB").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "ARC_BING_MB").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "GE").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "GE").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "SRTM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "SRTM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "ARC_BING_MB").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "ARC_BING_MB").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "GE").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "GE").c_str());
                        }
                        ImageCut((Input_Dir + "\\" + files[i]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + Image_Type + "\\" + files[i]).c_str(), jj * dx, ii * dy, (nImgSizeX - (Cut_Num_Row_Col - 1) * dx), dy, "GTiff");
                    }
                    else if (ii == Cut_Num_Row_Col - 1)
                    {
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "ARC_BING_MB").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "ARC_BING_MB").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "GE").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "GE").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "SRTM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "SRTM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "ARC_BING_MB").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "ARC_BING_MB").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "GE").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "GE").c_str());
                        }
                        ImageCut((Input_Dir + "\\" + files[i]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + Image_Type + "\\" + files[i]).c_str(), jj * dx, ii * dy, dx, (nImgSizeY - (Cut_Num_Row_Col - 1) * dy), "GTiff");
                    }
                    else
                    {
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "ARC_BING_MB").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "ARC_BING_MB").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "GE").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "GE").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "SRTM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + "SRTM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "ARC_BING_MB").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "ARC_BING_MB").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "GE").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "MATCH_RESULT" + "\\" + "GE").c_str());
                        }
                        ImageCut((Input_Dir + "\\" + files[i]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP" + "\\" + Image_Type + "\\" + files[i]).c_str(), jj * dx, ii * dy, dx, dy, "GTiff");
                    }
                }
            }

        }
        //GDALClose((GDALDatasetH)poDataset);
    }

    if (Image_Type == "DOM_DSM")
    {
        for (int i = 0; i < Cut_Num; i++)
        {
            string DOM_path = Input_Dir + "\\DOM";
            vector <string> DOM_file;
            GetAllFiles_tif(DOM_path, DOM_file);

            GDALAllRegister();//注册所有的驱动
            CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名

            //1、加载tif数据
            string file_path_name_dom = Input_Dir + "\\DOM\\" + DOM_file[0];

            GDALDataset* poDataset = (GDALDataset*)GDALOpen(file_path_name_dom.c_str(), GA_Update);//GA_Update和GA_ReadOnly两种模式
            if (poDataset == NULL)
                std::cout << "指定的文件不能打开!" << std::endl;

            //获取图像的尺寸
            int nImgSizeX = poDataset->GetRasterXSize();
            int nImgSizeY = poDataset->GetRasterYSize();

            int dx = nImgSizeX / sqrt(Cut_Num);
            int dy = nImgSizeY / sqrt(Cut_Num);

            double dom_trans[6] = { 0 };
            string dom_tfw_path = DOM_path + "\\orthophoto.tfw";
            FILE* p_dom;
            p_dom = fopen(dom_tfw_path.c_str(), "r");
            fscanf(p_dom, "%lf", &dom_trans[1]);
            fscanf(p_dom, "%lf", &dom_trans[2]);
            fscanf(p_dom, "%lf", &dom_trans[4]);
            fscanf(p_dom, "%lf", &dom_trans[5]);
            fscanf(p_dom, "%lf", &dom_trans[0]);
            fscanf(p_dom, "%lf", &dom_trans[3]);
            poDataset->SetGeoTransform(dom_trans);
            fclose(p_dom);

            int Cut_Num_Row_Col = sqrt(Cut_Num);
            for (int ii = 0; ii < Cut_Num_Row_Col; ii++)//按行排序
            {
                for (int jj = 0; jj < Cut_Num_Row_Col; jj++)
                {
                    if (jj == Cut_Num_Row_Col - 1 && ii == Cut_Num_Row_Col - 1)
                    {
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str());
                        }
                        ImageCut((Input_Dir + "\\DOM\\" + DOM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM" + "\\" + DOM_file[0]).c_str(), jj * dx, ii * dy, (nImgSizeX - (Cut_Num_Row_Col - 1) * dx), (nImgSizeY - (Cut_Num_Row_Col - 1) * dy), "GTiff");
                        ImageCut((Input_Dir + "\\DOM\\" + DOM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP\\ARC_BING_MB\\" + DOM_file[0]).c_str(), jj * dx, ii * dy, (nImgSizeX - (Cut_Num_Row_Col - 1) * dx), (nImgSizeY - (Cut_Num_Row_Col - 1) * dy), "GTiff");
                        ImageCut((Input_Dir + "\\DOM\\" + DOM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP\\GE\\" + DOM_file[0]).c_str(), jj * dx, ii * dy, (nImgSizeX - (Cut_Num_Row_Col - 1) * dx), (nImgSizeY - (Cut_Num_Row_Col - 1) * dy), "GTiff");
                    }
                    else if (jj == Cut_Num_Row_Col - 1)
                    {
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str());
                        }
                        ImageCut((Input_Dir + "\\DOM\\" + DOM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM" + "\\" + DOM_file[0]).c_str(), jj * dx, ii * dy, (nImgSizeX - (Cut_Num_Row_Col - 1) * dx), dy, "GTiff");
                        ImageCut((Input_Dir + "\\DOM\\" + DOM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP\\ARC_BING_MB\\" + DOM_file[0]).c_str(), jj * dx, ii * dy, (nImgSizeX - (Cut_Num_Row_Col - 1) * dx), dy, "GTiff");
                        ImageCut((Input_Dir + "\\DOM\\" + DOM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP\\GE\\" + DOM_file[0]).c_str(), jj * dx, ii * dy, (nImgSizeX - (Cut_Num_Row_Col - 1) * dx), dy, "GTiff");
                    }
                    else if (ii == Cut_Num_Row_Col - 1)
                    {
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str());
                        }
                        ImageCut((Input_Dir + "\\DOM\\" + DOM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM" + "\\" + DOM_file[0]).c_str(), jj * dx, ii * dy, dx, (nImgSizeY - (Cut_Num_Row_Col - 1) * dy), "GTiff");
                        ImageCut((Input_Dir + "\\DOM\\" + DOM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP\\ARC_BING_MB\\" + DOM_file[0]).c_str(), jj * dx, ii * dy, dx, (nImgSizeY - (Cut_Num_Row_Col - 1) * dy), "GTiff");
                        ImageCut((Input_Dir + "\\DOM\\" + DOM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP\\GE\\" + DOM_file[0]).c_str(), jj * dx, ii * dy, dx, (nImgSizeY - (Cut_Num_Row_Col - 1) * dy), "GTiff");
                    }
                    else
                    {
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj)).c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM").c_str());
                        }
                        if (0 != _access((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str(), 0))
                        {
                            _mkdir((Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM").c_str());
                        }
                        ImageCut((Input_Dir + "\\DOM\\" + DOM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DOM" + "\\" + DOM_file[0]).c_str(), jj * dx, ii * dy, dx, dy, "GTiff");
                        ImageCut((Input_Dir + "\\DOM\\" + DOM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP\\ARC_BING_MB\\" + DOM_file[0]).c_str(), jj * dx, ii * dy, dx, dy, "GTiff");
                        ImageCut((Input_Dir + "\\DOM\\" + DOM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "BASEMAP\\GE\\" + DOM_file[0]).c_str(), jj * dx, ii * dy, dx, dy, "GTiff");
                    }
                }
            }
            string DSM_path = Input_Dir + "\\DSM";
            vector <string> DSM_file;
            GetAllFiles_tif(DSM_path, DSM_file);

            GDALAllRegister();//注册所有的驱动
            CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名

            //1、加载tif数据
            string file_path_name_dsm = Input_Dir + "\\DSM\\" + DSM_file[0];

            GDALDataset* poDataset1 = (GDALDataset*)GDALOpen(file_path_name_dsm.c_str(), GA_Update);//GA_Update和GA_ReadOnly两种模式
            if (poDataset1 == NULL)
                std::cout << "指定的文件不能打开!" << std::endl;

            //获取图像的尺寸
            nImgSizeX = poDataset1->GetRasterXSize();
            nImgSizeY = poDataset1->GetRasterYSize();

            dx = nImgSizeX / sqrt(Cut_Num);
            dy = nImgSizeY / sqrt(Cut_Num);

            double dsm_trans[6] = { 0 };
            string dsm_tfw_path = DSM_path + "\\dsm_i.tfw";
            FILE* p_dsm;
            p_dsm = fopen(dsm_tfw_path.c_str(), "r");
            fscanf(p_dsm, "%lf", &dsm_trans[1]);
            fscanf(p_dsm, "%lf", &dsm_trans[2]);
            fscanf(p_dsm, "%lf", &dsm_trans[4]);
            fscanf(p_dsm, "%lf", &dsm_trans[5]);
            fscanf(p_dsm, "%lf", &dsm_trans[0]);
            fscanf(p_dsm, "%lf", &dsm_trans[3]);
            poDataset1->SetGeoTransform(dsm_trans);
            fclose(p_dsm);

            for (int ii = 0; ii < Cut_Num_Row_Col; ii++)//按行排序
            {
                for (int jj = 0; jj < Cut_Num_Row_Col; jj++)
                {
                    if (jj == Cut_Num_Row_Col - 1 && ii == Cut_Num_Row_Col - 1)
                    {
                        ImageCut((Input_Dir + "\\DSM\\" + DSM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM" + "\\" + DSM_file[0]).c_str(), jj* dx, ii* dy, (nImgSizeX - (Cut_Num_Row_Col - 1) * dx), (nImgSizeY - (Cut_Num_Row_Col - 1) * dy), "GTiff");
                        // ImageCut((Input_Dir + "\\DSM\\" + DSM_file[0]).c_str(), (Output_Dir + "\\" + to_string(i) + "\\" + "DSM" + "\\" + DSM_file[0]).c_str(), 0, 0, nImgSizeX, nImgSizeY, "GTiff");
                    }
                    else if (jj == Cut_Num_Row_Col - 1)
                    {
                        ImageCut((Input_Dir + "\\DSM\\" + DSM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM" + "\\" + DSM_file[0]).c_str(), jj* dx, ii* dy, (nImgSizeX - (Cut_Num_Row_Col - 1) * dx), dy, "GTiff");
                        // ImageCut((Input_Dir + "\\DSM\\" + DSM_file[0]).c_str(), (Output_Dir + "\\" + to_string(i) + "\\" + "DSM" + "\\" + DSM_file[0]).c_str(), 0, 0, nImgSizeX, nImgSizeY, "GTiff");
                    }
                    else if (ii == Cut_Num_Row_Col - 1)
                    {
                        ImageCut((Input_Dir + "\\DSM\\" + DSM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM" + "\\" + DSM_file[0]).c_str(), jj* dx, ii* dy, dx, (nImgSizeY - (Cut_Num_Row_Col - 1) * dy), "GTiff");
                        // ImageCut((Input_Dir + "\\DSM\\" + DSM_file[0]).c_str(), (Output_Dir + "\\" + to_string(i) + "\\" + "DSM" + "\\" + DSM_file[0]).c_str(), 0, 0, nImgSizeX, nImgSizeY, "GTiff");
                    }
                    else
                    {
                        ImageCut((Input_Dir + "\\DSM\\" + DSM_file[0]).c_str(), (Output_Dir + "\\" + to_string(ii * Cut_Num_Row_Col + jj) + "\\" + "DSM" + "\\" + DSM_file[0]).c_str(), jj * dx, ii * dy, dx, dy, "GTiff");
                        // ImageCut((Input_Dir + "\\DSM\\" + DSM_file[0]).c_str(), (Output_Dir + "\\" + to_string(i) + "\\" + "DSM" + "\\" + DSM_file[0]).c_str(), 0, 0, nImgSizeX, nImgSizeY, "GTiff");
                    }
                }
            }
        }
    }

}

void GetAllFiles_tif(string path, vector<string>& files)
{
    _int64 hFile = 0;
    //文件信息结构体
    struct _finddata_t fileinfo;
    string p;
    if ((hFile = _findfirst(p.assign(path).append("\\*.tif").c_str(), &fileinfo)) != -1)
    {
        do
        {
            files.push_back(fileinfo.name);
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}

void Image_Chunking_Process(string All_Area_Dir, int Cut_Num, string exe_path, bool is_Feature_Matching, bool is_Geometric_Correction, bool is_Precision_Evaluation, bool is_Building_Mask, string Mask_Type, bool is_Elevation_Weight)
{
    //第一步，将分块后的影像放入正确的文件夹中
    vector <string> Area_Folder;
    GetFolder(All_Area_Dir, Area_Folder);
    //if (Area_Folder.size() == 1)//需要从头开始，即把Guangzhou_is_Chunked去掉
    //{
    printf("正在进行分块操作\n");
    string path_ARC_BING_MB = All_Area_Dir + "\\" + Area_Folder[0] + "\\BASEMAP" + "\\ARC_BING_MB";
    string path_GE = All_Area_Dir + "\\" + Area_Folder[0] + "\\BASEMAP" + "\\GE";
    string path_SRTM = All_Area_Dir + "\\" + Area_Folder[0] + "\\BASEMAP" + "\\SRTM";
    if (0 != _access((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked").c_str(), 0))
    {
        // if this folder not exist, create a new one.
        _mkdir((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked").c_str());   // 返回 0 表示创建成功，-1 表示失败
    }
    Image_Chunking(path_ARC_BING_MB, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "ARC_BING_MB");
    Image_Chunking(path_GE, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "GE");
    Image_Chunking(path_SRTM, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "SRTM");// SRTM 不做裁剪
    //分块DOM和DSM，并把DOM放到对应的底图文件夹中
    string path_DOM_DSM = All_Area_Dir + "\\" + Area_Folder[0];
    Image_Chunking(path_DOM_DSM, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "DOM_DSM");

    printf("分块操作已完成\n");
    //}
    //else
    //{
    //    printf("无法完成底图分块操作，由于有一个以上文件夹处于数据所在路径下，默认已完成分块进行后续操作\n");
    //}

    //第二步，将底图BASEMAP文件夹中的两个文件夹中的文件写成分别命名的txt
    vector <string> Chunked_Folder;
    GetFolder(All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Chunked_Folder);
    if (is_Feature_Matching == true)
    {
        printf("正在进行特征匹配\n");
        for (int i = 0; i < Chunked_Folder.size(); i++)
        {
            vector <string> files1;
            string dir_path1 = All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\ARC_BING_MB";
            GetAllFiles_tif(dir_path1, files1);
            FILE* p1;
            p1 = fopen((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\ARC_BING_MB.txt").c_str(), "w");
            for (int j = 0; j < files1.size(); j++)
            {
                if (files1[j] != "orthophoto.tif")
                {
                    fprintf(p1, "orthophoto.tif %s\n", files1[j].c_str());
                }
            }
            fclose(p1);

            vector <string> files2;
            string dir_path2 = All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\GE";
            GetAllFiles_tif(dir_path2, files2);
            FILE* p2;
            p2 = fopen((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\GE.txt").c_str(), "w");
            for (int j = 0; j < files2.size(); j++)
            {
                if (files2[j] != "orthophoto.tif")
                {            
                    fprintf(p2, "orthophoto.tif %s\n", files2[j].c_str());
                }
            }
            fclose(p2);
        }


        //第三步，写bat文件，调用match_pair.exe,注意输入文件夹，输入文件的txt，输出路径
        for (int i = 0; i < Chunked_Folder.size(); i++)
        {
            FILE* p3;
            p3 = fopen((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\ARC_BING_MB_Match_Pair.bat").c_str(), "w");
            fprintf(p3, ("cmd /c \"cd /d " + exe_path + "&&" + exe_path + "/match_pairs.exe --resize -1 --superglue outdoor --max_keypoints -1 --keypoint_threshold 0.04 --nms_radius 3 --match_threshold 0.90 --resize_float --viz --viz_extension png --input_dir %s --input_pairs %s --output_dir %s\"").c_str(),
                (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\ARC_BING_MB/").c_str(), (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\ARC_BING_MB.txt").c_str(), (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\MATCH_RESULT\\ARC_BING_MB/").c_str());
            fclose(p3);
            FILE* p4;
            p4 = fopen((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\GE_Match_Pair.bat").c_str(), "w");
            fprintf(p4, ("cmd /c \"cd /d " + exe_path + "&&" + exe_path + "/match_pairs.exe --resize -1 --superglue outdoor --max_keypoints -1 --keypoint_threshold 0.04 --nms_radius 3 --match_threshold 0.90 --resize_float --viz --viz_extension png --input_dir %s --input_pairs %s --output_dir %s\"").c_str(),
                (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\GE/").c_str(), (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\GE.txt").c_str(), (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\MATCH_RESULT\\GE/").c_str());
            fclose(p4);

            system((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\ARC_BING_MB_Match_Pair.bat").c_str());
            system((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\GE_Match_Pair.bat").c_str());
        }
        printf("已完成特征匹配\n");
    }

    if (is_Geometric_Correction == true)
    {
        printf("正在进行DSM和DOM的几何校正......\n");
        //第四步，is_merged为true，改算法里面的循环；
        //vector <vector <Point3d>> BM_All_Points_ARC_BING_MB_All;
        vector <vector <Point3d>> BM_All_Points_GE_All;
        //vector <vector <Point3d>> BM_All_Points_DSM_ARC_BING_MB_All;
        vector <vector <Point3d>> BM_All_Points_DSM_GE_All;

        //这个for循环的目的是将所有分块区域的匹配点都加入到同一个vector数组中，从而可以达到和整体平差基本一样的结果
        for (int folder_area = 0; folder_area < Chunked_Folder.size(); folder_area++)
        {
            //vector <vector <Point3d>> BM_All_Points_ARC_BING_MB;
            //vector <vector <Point3d>> BM_All_Points_DSM_ARC_BING_MB;
            //GetAllFiles_txt2GeoInfo_BM_and_DSM(All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[folder_area], "ARC_BING_MB", BM_All_Points_ARC_BING_MB, BM_All_Points_DSM_ARC_BING_MB);
            //for (int i = 0; i < BM_All_Points_ARC_BING_MB.size(); i++)
            //    BM_All_Points_ARC_BING_MB_All.push_back(BM_All_Points_ARC_BING_MB[i]);
            //for (int i = 0; i < BM_All_Points_DSM_ARC_BING_MB.size(); i++)
            //    BM_All_Points_DSM_ARC_BING_MB_All.push_back(BM_All_Points_DSM_ARC_BING_MB[i]);
            vector <vector <Point3d>> BM_All_Points_GE;
            vector <vector <Point3d>> BM_All_Points_DSM_GE;
            GetAllFiles_txt2GeoInfo_BM_and_DSM(All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[folder_area], "GE", BM_All_Points_GE, BM_All_Points_DSM_GE);
            for (int i = 0; i < BM_All_Points_GE.size(); i++)
                BM_All_Points_GE_All.push_back(BM_All_Points_GE[i]);
            for (int i = 0; i < BM_All_Points_DSM_GE.size(); i++)
                BM_All_Points_DSM_GE_All.push_back(BM_All_Points_DSM_GE[i]);
        }

        removeNaNPoints(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All);

        Point3d Model_Translation_Parameter_ARC_BING_MB_Ransac;
        Point3d Model_Translation_Parameter_GE_Ransac;
        Point3d Model_Translation_Parameter_ARC_BING_MB_Iter;
        Point3d Model_Translation_Parameter_GE_Iter;

        //printf("正在进行ARC_BING_MB的粗校正......\n");
        //RANSAC_Point(BM_All_Points_ARC_BING_MB_All, BM_All_Points_DSM_ARC_BING_MB_All, 6000, RANSAC_THRESHOLD, 0.995, RANSAC_RATIO, Model_Translation_Parameter_ARC_BING_MB_Ransac, string("ARC_BING_MB"), All_Area_Dir + "\\" + Area_Folder[0]);
        //double x_ARC_BING_MB = Model_Translation_Parameter_ARC_BING_MB_Ransac.x;
        //double y_ARC_BING_MB = Model_Translation_Parameter_ARC_BING_MB_Ransac.y;
        //double z_ARC_BING_MB = Model_Translation_Parameter_ARC_BING_MB_Ransac.z;
        //printf("已完成ARC_BING_MB的粗校正\n");

        //if (is_Building_Mask == true)
        //{
        //    printf("正在进行基于%s建筑物掩膜的粗差剔除......\n", Mask_Type.c_str());
        //    string Mask_Path = All_Area_Dir + "\\" + Area_Folder[0] + "\\Building_Mask";
        //    Building_Mask_Elimination_Cross_Error(BM_All_Points_ARC_BING_MB_All, BM_All_Points_DSM_ARC_BING_MB_All, Mask_Path, Mask_Type, "ARC_BING_MB", Cut_Num);
        //    printf("已完成基于%s建筑物掩膜的粗差剔除\n", Mask_Type.c_str());
        //}

        //printf("正在进行ARC_BING_MB的精校正......\n");
        //if (is_Elevation_Weight == true)
        //{
        //    // 计算高程的权重（置信度）
        //    printf("正在进行高程赋权和模型参数z的重新计算......\n");
        //    string Mask_Path = All_Area_Dir + "\\" + Area_Folder[0] + "\\Building_Mask";
        //    vector <vector<double>> finalConfidenceMatrix1;
        //    Elevation_Weight_Calculation(BM_All_Points_ARC_BING_MB_All, BM_All_Points_DSM_ARC_BING_MB_All, Mask_Path, "ARC_BING_MB", All_Area_Dir + "\\" + Area_Folder[0], finalConfidenceMatrix1);
        //    printf("已完成高程赋权和模型参数z的更新\n");
        //    Weight_Iteration_Point_Elevation(BM_All_Points_ARC_BING_MB_All, BM_All_Points_DSM_ARC_BING_MB_All, x_ARC_BING_MB, y_ARC_BING_MB, z_ARC_BING_MB, WEIGHT_ITERATION_SIGMA_BEF, Model_Translation_Parameter_ARC_BING_MB_Iter, finalConfidenceMatrix1, All_Area_Dir + "\\" + Area_Folder[0]);
        //}
        //else
        //{
        //    Weight_Iteration_Point(BM_All_Points_ARC_BING_MB_All, BM_All_Points_DSM_ARC_BING_MB_All, x_ARC_BING_MB, y_ARC_BING_MB, z_ARC_BING_MB, WEIGHT_ITERATION_SIGMA_BEF, Model_Translation_Parameter_ARC_BING_MB_Iter);
        //}
        //double x_ARC_BING_MB_iter = Model_Translation_Parameter_ARC_BING_MB_Iter.x;
        //double y_ARC_BING_MB_iter = Model_Translation_Parameter_ARC_BING_MB_Iter.y;
        //double z_ARC_BING_MB_iter = Model_Translation_Parameter_ARC_BING_MB_Iter.z;
        //printf("已完成ARC_BING_MB的精校正\n");

        printf("正在进行GE的粗校正......\n");

        string Mask_Prob_Path = All_Area_Dir + "\\" + Area_Folder[0] + "\\Building_Mask_Prob";

        vector <vector<float>> Building_Prob_Weight;
        // 遍历BM_All_Points_GE_All来定义Building_Prob_Weight的结构
        for (const auto& subVector : BM_All_Points_GE_All) {
            // 使用subVector的大小来创建一个float向量
            Building_Prob_Weight.push_back(std::vector<float>(subVector.size(), 0.0f));  // 初始化为0.0f
        }

        RANSAC_Point_Semantic_Constraint(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All, Building_Prob_Weight, 6000, RANSAC_THRESHOLD, 0.995, RANSAC_RATIO, Model_Translation_Parameter_GE_Ransac, string("GE"), All_Area_Dir + "\\" + Area_Folder[0], Mask_Prob_Path, Mask_Type, "GE", Cut_Num);
        
        // RANSAC_Point(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All, 6000, RANSAC_THRESHOLD, 0.995, RANSAC_RATIO, Model_Translation_Parameter_GE_Ransac, string("GE"), All_Area_Dir + "\\" + Area_Folder[0]);
        double x_GE = Model_Translation_Parameter_GE_Ransac.x;
        double y_GE = Model_Translation_Parameter_GE_Ransac.y;
        double z_GE = Model_Translation_Parameter_GE_Ransac.z;
        printf("已完成GE的粗校正\n");

        Point3d Model_Params;
        Model_Params.x = x_GE;
        Model_Params.y = y_GE;
        Model_Params.z = z_GE;

        calculateGeoPositioningAccuracy("Coarse Correction", Model_Params, All_Area_Dir + "\\Truth_GCP.txt", All_Area_Dir + "\\Unrectified_GCP.txt");


        if (is_Building_Mask == true)
        {
            printf("正在进行基于%s建筑物掩膜的粗差剔除......\n", Mask_Type.c_str());
            string Mask_Path = All_Area_Dir + "\\" + Area_Folder[0] + "\\Building_Mask";
            Building_Mask_Elimination_Cross_Error(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All, Mask_Path, Mask_Type, "GE", Cut_Num);
            printf("已完成基于%s建筑物掩膜的粗差剔除\n", Mask_Type.c_str());
        }

        printf("正在进行GE的精校正......\n");
        if (is_Elevation_Weight == true)
        {
            // 计算高程的权重（置信度）
            printf("正在进行高程赋权和模型参数z的重新计算......\n");
            string Mask_Path = All_Area_Dir + "\\" + Area_Folder[0] + "\\Building_Mask";
            vector <vector<double>> finalConfidenceMatrix2;
            Elevation_Weight_Calculation(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All, Mask_Path, "GE", All_Area_Dir + "\\" + Area_Folder[0], finalConfidenceMatrix2);
            printf("已完成高程赋权和模型参数z的更新\n");
            Weight_Iteration_Point_Elevation(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All, x_GE, y_GE, z_GE, WEIGHT_ITERATION_SIGMA_BEF, Model_Translation_Parameter_GE_Iter, finalConfidenceMatrix2, All_Area_Dir + "\\" + Area_Folder[0]);
        }
        else
        {
            Weight_Iteration_Point(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All, x_GE, y_GE, z_GE, WEIGHT_ITERATION_SIGMA_BEF, Model_Translation_Parameter_GE_Iter);
        }
        double x_GE_iter = Model_Translation_Parameter_GE_Iter.x;
        double y_GE_iter = Model_Translation_Parameter_GE_Iter.y;
        double z_GE_iter = Model_Translation_Parameter_GE_Iter.z;
        printf("已完成GE的精校正\n");

        Model_Params.x = x_GE_iter;
        Model_Params.y = y_GE_iter;
        Model_Params.z = z_GE_iter;

        calculateGeoPositioningAccuracy("Coarse+Fine Correction", Model_Params, All_Area_Dir + "\\Truth_GCP.txt", All_Area_Dir + "\\Unrectified_GCP.txt");


        //printf("ARC_BING_MB:\nArea_Folder     x      y     z\n");
        //printf("%s %lf   %lf   %lf\n", Area_Folder[0].c_str(), Model_Translation_Parameter_ARC_BING_MB_Ransac.x, Model_Translation_Parameter_ARC_BING_MB_Ransac.y, Model_Translation_Parameter_ARC_BING_MB_Ransac.z);
        //printf("ARC_BING_MB_Iter:\nArea_Folder     x      y     z\n");
        //printf("%s %lf   %lf   %lf\n", Area_Folder[0].c_str(), Model_Translation_Parameter_ARC_BING_MB_Iter.x, Model_Translation_Parameter_ARC_BING_MB_Iter.y, Model_Translation_Parameter_ARC_BING_MB_Iter.z);
        printf("Google_Earth:\nArea_Folder     x      y     z\n");
        printf("%s %lf   %lf   %lf\n", Area_Folder[0].c_str(), Model_Translation_Parameter_GE_Ransac.x, Model_Translation_Parameter_GE_Ransac.y, Model_Translation_Parameter_GE_Ransac.z);
        printf("Google_Earth_Iter:\nArea_Folder     x      y     z\n");
        printf("%s %lf   %lf   %lf\n", Area_Folder[0].c_str(), Model_Translation_Parameter_GE_Iter.x, Model_Translation_Parameter_GE_Iter.y, Model_Translation_Parameter_GE_Iter.z);
        string txt_path = All_Area_Dir + "\\Model_Translate_Parameters.txt";
        FILE* p_Model;
        p_Model = fopen(txt_path.c_str(), "w");
        //fprintf(p_Model, "ARC_BING_MB:\nArea_Folder     x      y     z\n");
        //fprintf(p_Model, "%s %lf   %lf   %lf\n", All_Area_Dir.c_str(), Model_Translation_Parameter_ARC_BING_MB_Ransac.x, Model_Translation_Parameter_ARC_BING_MB_Ransac.y, Model_Translation_Parameter_ARC_BING_MB_Ransac.z);
        //fprintf(p_Model, "ARC_BING_MB_Iter:\nArea_Folder     x      y     z\n");
        //fprintf(p_Model, "%s %lf   %lf   %lf\n", All_Area_Dir.c_str(), Model_Translation_Parameter_ARC_BING_MB_Iter.x, Model_Translation_Parameter_ARC_BING_MB_Iter.y, Model_Translation_Parameter_ARC_BING_MB_Iter.z);
        fprintf(p_Model, "Google_Earth:\nArea_Folder     x      y     z\n");
        fprintf(p_Model, "%s %lf   %lf   %lf\n", All_Area_Dir.c_str(), Model_Translation_Parameter_GE_Ransac.x, Model_Translation_Parameter_GE_Ransac.y, Model_Translation_Parameter_GE_Ransac.z);
        fprintf(p_Model, "Google_Earth_Iter:\nArea_Folder     x      y     z\n");
        fprintf(p_Model, "%s %lf   %lf   %lf\n", All_Area_Dir.c_str(), Model_Translation_Parameter_GE_Iter.x, Model_Translation_Parameter_GE_Iter.y, Model_Translation_Parameter_GE_Iter.z);
        fclose(p_Model);

        //第五步，根据模型参数校正DOM和DSM
        if (0 != _access((All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "Rectified_DOM_DSM").c_str(), 0))
        {
            _mkdir((All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "Rectified_DOM_DSM").c_str());   // 返回 0 表示创建成功，-1 表示失败
        }
        string dom_path_tif = All_Area_Dir + "\\" + Area_Folder[0] + "\\DOM\\orthophoto.tif";
        string dsm_path_tif = All_Area_Dir + "\\" + Area_Folder[0] + "\\DSM\\dsm_i.tif";

        //string dom_rect_path_tif_A_iter = All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "Rectified_DOM_DSM\\orthophoto_Rect_ARC_BING_MB.tif";
        //DOM_Rectified(dom_path_tif.c_str(), dom_rect_path_tif_A_iter.c_str(), "GTiff", Model_Translation_Parameter_ARC_BING_MB_Iter.x, Model_Translation_Parameter_ARC_BING_MB_Iter.y);
        //string dsm_rect_path_tif_A_iter = All_Area_Dir + "\\" + Area_Folder[0] + "\\Rectified_DOM_DSM\\DSM_Rect_ARC_BING_MB.tif";
        ////实现DSM的z方向校正
        //DSM_Rectified(dsm_path_tif.c_str(), dsm_rect_path_tif_A_iter.c_str(), "GTiff", Model_Translation_Parameter_ARC_BING_MB_Iter.x, Model_Translation_Parameter_ARC_BING_MB_Iter.y, Model_Translation_Parameter_ARC_BING_MB_Iter.z);

        string dom_rect_path_tif_GE_iter = All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "Rectified_DOM_DSM\\orthophoto_Rect_GE.tif";
        DOM_Rectified(dom_path_tif.c_str(), dom_rect_path_tif_GE_iter.c_str(), "GTiff", Model_Translation_Parameter_GE_Iter.x, Model_Translation_Parameter_GE_Iter.y);
        string dsm_rect_path_tif_GE_iter = All_Area_Dir + "\\" + Area_Folder[0] + "\\Rectified_DOM_DSM\\DSM_Rect_GE.tif";
        //实现DSM的z方向校正
        DSM_Rectified(dsm_path_tif.c_str(), dsm_rect_path_tif_GE_iter.c_str(), "GTiff", Model_Translation_Parameter_GE_Iter.x, Model_Translation_Parameter_GE_Iter.y, Model_Translation_Parameter_GE_Iter.z);
        printf("已完成DSM和DOM的几何校正\n");
    }


    //精度评价
    if (is_Precision_Evaluation == true)
    {
        printf("正在对几何校正后的DSM进行精度评价......\n");

        string TruthFile = All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "TRUTH";
        vector <string> Truth_tif;
        GetAllFiles_tif(TruthFile, Truth_tif);

        string RectFile = All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "Rectified_DOM_DSM";
        vector <string> Rect_tif;
        GetAllFiles_tif(RectFile, Rect_tif);

        string UnRectFile = All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "DSM";
        vector <string> UnRect_tif;
        GetAllFiles_tif(UnRectFile, UnRect_tif);

        //去除校正后DOM的影响，因为精度评价仅需要DSM
        for (int j = 0; j < Rect_tif.size(); j++)
        {
            if (Rect_tif[j].find("DSM") == string::npos)
            {
                Rect_tif.erase(Rect_tif.begin() + j);
                j--;
            }
        }

        //创建文件夹存放精度评价矩阵
        if (0 != _access((All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation").c_str(), 0))
        {
            _mkdir((All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation").c_str());   // 返回 0 表示创建成功，-1 表示失败
        }
        if (0 != _access((All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation_UnRect").c_str(), 0))
        {
            _mkdir((All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation_UnRect").c_str());   // 返回 0 表示创建成功，-1 表示失败
        }

        //对校正后的DSM进行精度评价
        for (int j = 0; j < Rect_tif.size(); j++)
        {
            Precision_Evaluation((TruthFile + "\\" + Truth_tif[0]).c_str(), (RectFile + "\\" + Rect_tif[j]).c_str(), All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation", Area_Folder[0], false);
        }
        //对未校正的DSM进行精度评价
        Precision_Evaluation((TruthFile + "\\" + Truth_tif[0]).c_str(), (UnRectFile + "\\" + UnRect_tif[0]).c_str(), All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation_UnRect", Area_Folder[0], false);

        printf("已完成对几何校正后DSM的精度评价\n");
    }
}


void GetFolder(string path, vector<string>& folder)
{
    long long hFile = 0;     //注意一定要用long long
    struct _finddata_t fileInfo;
    string p;
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileInfo)) != -1)
    {
        do
        {
            //如果是目录，存入列表
            if ((fileInfo.attrib & _A_SUBDIR))
            {

                if (strcmp(fileInfo.name, ".") != 0 && strcmp(fileInfo.name, "..") != 0)
                    folder.push_back(fileInfo.name);
            }
            else
            {
                continue;

            }
        } while (_findnext(hFile, &fileInfo) == 0);
    }
    _findclose(hFile);
}


void GetAllFiles_tfw(string path, vector<string>& files)
{
    _int64 hFile = 0;
    //文件信息结构体
    struct _finddata_t fileinfo;
    string p;
    if ((hFile = _findfirst(p.assign(path).append("\\*.tfw").c_str(), &fileinfo)) != -1)
    {
        do
        {
            files.push_back(fileinfo.name);
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}


void RANSAC_Point(vector <vector<Point3d>>& BM_pt, vector <vector<Point3d>>& DSM_pt, int Init_Iter_Time, double Threshold, double confidence, double Inlier_Ratio, Point3d &Model_Translation_Parameter, string BM_name, string dirpath)
{
    int all_bef_RANSAC = 0;
    for (int ri = 0; ri < BM_pt.size(); ri++)
    {
        all_bef_RANSAC += BM_pt[ri].size();
    }
    printf("RANSAC剔除前点数 %d", all_bef_RANSAC);

    // 创建RANSAC前后点文件夹
    if (0 != _access((dirpath + "\\" + "RANSAC_pt_res").c_str(), 0))
    {
        int result = _mkdir((dirpath + "\\" + "RANSAC_pt_res").c_str());   // 返回 0 表示创建成功，-1 表示失败
        if (result != 0) {
            switch (errno) {
            case ENOENT:
                std::cerr << "路径错误！\n" << std::endl;
                break;
            default:
                std::cerr << "创建文件夹时遇到未知错误，请检查文件夹权限！\n" << std::endl;
                break;
            }
        }
    }

    // 输出一个RANSAC前的所有DSM_pt点坐标的txt文件，此处为地理位置的x，y
    FILE* p_bef;
    string RANSAC_txt_bef = dirpath + "\\RANSAC_pt_res\\RANSAC_bef_" + BM_name + ".txt";
    p_bef = fopen(RANSAC_txt_bef.c_str(), "w");

    for (int ii = 0; ii < BM_pt.size(); ii++)
    {
        for (int jj = 0; jj < BM_pt[ii].size(); jj++)
        {
            if (isnan(DSM_pt[ii][jj].z) || isnan(BM_pt[ii][jj].z))
            {
                BM_pt[ii].erase(BM_pt[ii].begin() + jj);
                DSM_pt[ii].erase(DSM_pt[ii].begin() + jj);
                jj--;
                continue;
            }
            fprintf(p_bef, "%lf %lf\n", DSM_pt[ii][jj].x, DSM_pt[ii][jj].y);
        }
    }
    fclose(p_bef);

    // 为了避免局部匹配点质量一般导致校正参数被带偏，只使用整体的点进行模型校正参数的估算
    //for (int match_num = 0; match_num < BM_pt.size(); match_num++)
    //{
    //    int iter_time = Init_Iter_Time;
    //    int Max_Count = DSM_pt[match_num].size();
    //    if (Max_Count == 0)
    //    {
    //        continue;
    //    }
    //    int Good_Model_key = 0;
    //    int Good_Model_Count = 0;
    //    double Model_Para[3];//依次为x，y，z的平移量，为BM_pt - DSM_pt(参考真值减观测值)
    //    for (int i = 0; i < iter_time; i++)
    //    {
    //        //随机取点，计算平移模型参数
    //        int rand_num = (int)(round(1.0 * rand() / RAND_MAX * (Max_Count - 1)));

    //        Model_Para[0] = BM_pt[match_num][rand_num].x - DSM_pt[match_num][rand_num].x;
    //        Model_Para[1] = BM_pt[match_num][rand_num].y - DSM_pt[match_num][rand_num].y;
    //        Model_Para[2] = BM_pt[match_num][rand_num].z - DSM_pt[match_num][rand_num].z;
    //        //计算内点数量
    //        int inlier_count = 0;
    //        for (int ii = 0; ii < Max_Count; ii++)
    //        {
    //            double temp_x = DSM_pt[match_num][ii].x + Model_Para[0];
    //            double temp_y = DSM_pt[match_num][ii].y + Model_Para[1];
    //            double temp_z = DSM_pt[match_num][ii].z + Model_Para[2];
    //            double dx = abs(temp_x - BM_pt[match_num][ii].x);
    //            double dy = abs(temp_y - BM_pt[match_num][ii].y);
    //            double dz = abs(temp_z - BM_pt[match_num][ii].z);

    //            double dist = sqrt(dx * dx + dy * dy + dz * dz);

    //            if (dist < Threshold) inlier_count++;
    //        }
    //        if (inlier_count > round(Max_Count * Inlier_Ratio))//更新迭代次数
    //        {
    //            double p = double(inlier_count) / Max_Count;
    //            double numerator = log(1 - confidence);
    //            double deminator = log(1 - p);//模型仅需一个点，故为p的一次方
    //            double temp_iter_time = ceil(numerator / deminator);
    //            if (temp_iter_time <= i)
    //            {
    //                iter_time = int(ceil(i + 1));
    //            }
    //            else iter_time = int(ceil(temp_iter_time));
    //        }
    //        //避免没有好点导致自动选第一个的情况
    //        if (inlier_count > Good_Model_Count)
    //        {
    //            Good_Model_Count = inlier_count;
    //            Good_Model_key = rand_num;
    //        }
    //    }
    //    Model_Para[0] = BM_pt[match_num][Good_Model_key].x - DSM_pt[match_num][Good_Model_key].x;
    //    Model_Para[1] = BM_pt[match_num][Good_Model_key].y - DSM_pt[match_num][Good_Model_key].y;
    //    Model_Para[2] = BM_pt[match_num][Good_Model_key].z - DSM_pt[match_num][Good_Model_key].z;
    //    //删除外点，BM_pt和DSM_pt两个点集都删除
    //    for (int ii = 0; ii < BM_pt[match_num].size(); ii++)
    //    {
    //        double temp_x = DSM_pt[match_num][ii].x + Model_Para[0];
    //        double temp_y = DSM_pt[match_num][ii].y + Model_Para[1];
    //        double temp_z = DSM_pt[match_num][ii].z + Model_Para[2];
    //        double dx = abs(temp_x - BM_pt[match_num][ii].x);
    //        double dy = abs(temp_y - BM_pt[match_num][ii].y);
    //        double dz = abs(temp_z - BM_pt[match_num][ii].z);

    //        double dist = sqrt(dx * dx + dy * dy + dz * dz);

    //        //删除外点
    //        if (dist > Threshold)
    //        {
    //            BM_pt[match_num].erase(BM_pt[match_num].begin() + ii);
    //            DSM_pt[match_num].erase(DSM_pt[match_num].begin() + ii);
    //            ii--;
    //        }
    //    }
    //}

    //所有匹配对的整体RANSAC模型
    int iter_time = Init_Iter_Time;
    int Max_Count = 0;
    for (int i = 0; i < BM_pt.size(); i++)
    {
        Max_Count += BM_pt[i].size();
    }
    int Good_Model_key1 = 0;
    int Good_Model_key2 = 0;
    int Good_Model_Count = 0;
    double Model_Para[3];//依次为x，y，z的平移量，为BM_pt - DSM_pt(参考真值减观测值)

    for (int i = 0; i < iter_time; i++)
    {
        // TODO: 生成随机点
        //随机取点，计算平移模型参数
        int rand_num1 = generateRandomNumber(BM_pt.size());//注意erase的时候不要把整个匹配对都删除了
        if (BM_pt[rand_num1].size() == 0)
        {
            i--;
            continue;
        }
        int rand_num2 = generateRandomNumber(BM_pt[rand_num1].size());

        Model_Para[0] = BM_pt[rand_num1][rand_num2].x - DSM_pt[rand_num1][rand_num2].x;
        Model_Para[1] = BM_pt[rand_num1][rand_num2].y - DSM_pt[rand_num1][rand_num2].y;
        Model_Para[2] = BM_pt[rand_num1][rand_num2].z - DSM_pt[rand_num1][rand_num2].z;

        //计算内点数量
        int inlier_count = 0;
        for (int ii = 0; ii < BM_pt.size(); ii++)
        {
            for (int jj = 0; jj < BM_pt[ii].size(); jj++)
            {
                double temp_x = DSM_pt[ii][jj].x + Model_Para[0];
                double temp_y = DSM_pt[ii][jj].y + Model_Para[1];
                double temp_z = DSM_pt[ii][jj].z + Model_Para[2];
                double dx = abs(temp_x - BM_pt[ii][jj].x);
                double dy = abs(temp_y - BM_pt[ii][jj].y);
                double dz = abs(temp_z - BM_pt[ii][jj].z);

                double dist = sqrt(dx * dx + dy * dy + dz * dz);

                if (dist < Threshold) inlier_count++;
            }
        }

        if (inlier_count > round(Max_Count * Inlier_Ratio))//更新迭代次数
        {
            double p = double(inlier_count) / Max_Count;
            double numerator = log(1 - confidence);
            double deminator = log(1 - p);//模型仅需一个点，故为p的一次方
            double temp_iter_time = ceil(numerator / deminator);
            if (temp_iter_time <= i)
            {
                iter_time = int(ceil(i + temp_iter_time));
            }
            else iter_time = int(ceil(temp_iter_time));
        }
        //2023.1.13修改，为避免没有满足比例阈值直接选第一个点对的情况
        if (inlier_count > Good_Model_Count)
        {
            Good_Model_Count = inlier_count;
            Good_Model_key1 = rand_num1;
            Good_Model_key2 = rand_num2;
        }
    }
    Model_Para[0] = BM_pt[Good_Model_key1][Good_Model_key2].x - DSM_pt[Good_Model_key1][Good_Model_key2].x;
    Model_Para[1] = BM_pt[Good_Model_key1][Good_Model_key2].y - DSM_pt[Good_Model_key1][Good_Model_key2].y;
    Model_Para[2] = BM_pt[Good_Model_key1][Good_Model_key2].z - DSM_pt[Good_Model_key1][Good_Model_key2].z;
    //删除外点，两个点集都删除
    for (int ii = 0; ii < BM_pt.size(); ii++)
    {
        for (int jj = 0; jj < BM_pt[ii].size(); jj++)
        {
            double temp_x = DSM_pt[ii][jj].x + Model_Para[0];
            double temp_y = DSM_pt[ii][jj].y + Model_Para[1];
            double temp_z = DSM_pt[ii][jj].z + Model_Para[2];
            double dx = abs(temp_x - BM_pt[ii][jj].x);
            double dy = abs(temp_y - BM_pt[ii][jj].y);
            double dz = abs(temp_z - BM_pt[ii][jj].z);

            double dist = sqrt(dx * dx + dy * dy + dz * dz);
            //删除外点
            if (dist > Threshold)
            {
                BM_pt[ii].erase(BM_pt[ii].begin() + jj);
                DSM_pt[ii].erase(DSM_pt[ii].begin() + jj);
                jj--;
            }
        }
    }

    int total_inlier_num = 0;
    for (int i = 0; i < BM_pt.size(); i++)
    {
        total_inlier_num += BM_pt[i].size();
    }
    printf(" RANSAC剔除后点数 %d ", total_inlier_num);
    //printf("内外点比例为：%lf\n", float(total_inlier_num) / all_bef_RANSAC);

    Model_Para[0] = 0;
    Model_Para[1] = 0;
    Model_Para[2] = 0;

    // 这段出问题了
    for (int ii = 0; ii < BM_pt.size(); ii++)
    {
         for (int jj = 0; jj < BM_pt[ii].size(); jj++)
        {
            Model_Para[0] += BM_pt[ii][jj].x - DSM_pt[ii][jj].x;
            Model_Para[1] += BM_pt[ii][jj].y - DSM_pt[ii][jj].y;
            Model_Para[2] += BM_pt[ii][jj].z - DSM_pt[ii][jj].z;
        }
    }

    // 输出一个RANSAC后的所有DSM_pt点坐标的txt文件，此处为地理位置的x，y
    FILE* p_aft;
    string RANSAC_txt_aft = dirpath + "\\RANSAC_pt_res\\RANSAC_aft_" + BM_name + ".txt";
    p_aft = fopen(RANSAC_txt_aft.c_str(), "w");

    for (int ii = 0; ii < BM_pt.size(); ii++)
    {
        for (int jj = 0; jj < BM_pt[ii].size(); jj++)
        {
            fprintf(p_aft, "%lf %lf\n", DSM_pt[ii][jj].x, DSM_pt[ii][jj].y);
        }
    }
    fclose(p_aft);

    Model_Para[0] /= total_inlier_num;
    Model_Para[1] /= total_inlier_num;
    Model_Para[2] /= total_inlier_num;
    Point3d Model_Para_tmp;
    Model_Para_tmp.x = Model_Para[0];
    Model_Para_tmp.y = Model_Para[1];
    Model_Para_tmp.z = Model_Para[2];
    Model_Translation_Parameter = Model_Para_tmp;
}


void Weight_Iteration_Point(vector <vector<Point3d>> BM_pt, vector <vector<Point3d>> DSM_pt, double Init_Para_x, double Init_Para_y, double Init_Para_z, double sigma_bef, Point3d &Model_Translation_Parameter_WEIGHT_ITER)
{
    double Model_Para[3];
    Model_Para[0] = Init_Para_x;
    Model_Para[1] = Init_Para_y;
    Model_Para[2] = Init_Para_z;

    // double V_sum;
    int match_num = BM_pt.size();//外层控制匹配对数目
    int iter_time = 0;
    vector<vector <double>> P_last;

    while (1)
    {
        iter_time++;
        double pt_total_P = 0;
        double x_update = 0;
        double y_update = 0;
        double z_update = 0;

        for (int i = 0; i < match_num; i++)//外层控制，匹配对数目
        {
            int match_pt_num = BM_pt[i].size();//内层控制，各个匹配对中点的数目
            vector <double> P_last_tmp;
            for (int j = 0; j < match_pt_num; j++)
            {
                //计算当前模型下的点校正后坐标，与底图控制点坐标之间的差距即为V_dist
                double x_rect = DSM_pt[i][j].x + Model_Para[0];
                double y_rect = DSM_pt[i][j].y + Model_Para[1];
                double z_rect = DSM_pt[i][j].z + Model_Para[2];
                double dx = abs(BM_pt[i][j].x - x_rect);
                double dy = abs(BM_pt[i][j].y - y_rect);
                double dz = abs(BM_pt[i][j].z - z_rect);
                double V_dist = sqrt(dx * dx + dy * dy + dz * dz);
                //IGG3权函数，k0取2.5，k1取3.5
                double k0 = K_0, k1 = K_1;
                if (V_dist / sigma_bef < k0) // 可信段
                {
                    if (iter_time == 1)
                    {
                        P_last_tmp.push_back(1.0);
                    }
                }
                if (V_dist / sigma_bef >= k0 && V_dist / sigma_bef < k1) // 可疑段
                {
                    double d = (k1 - (V_dist / sigma_bef)) / (k1 - k0);
                    double w = d * d * k0 / (V_dist / sigma_bef);
                    if (iter_time == 1)
                    {
                        P_last_tmp.push_back(w);
                    }
                    else
                    {
                        P_last[i][j] = P_last[i][j] * w;//要记录上一次的权重
                    }
                }
                if (V_dist / sigma_bef >= k1) // 粗差段
                {
                    if (iter_time == 1)
                    {
                        P_last_tmp.push_back(0);
                    }
                    else
                    {
                        P_last[i][j] = 0;
                    }
                }
                //计算加权平均值
                if (iter_time == 1)
                {
                    pt_total_P += P_last_tmp[j];
                    x_update += (BM_pt[i][j].x - DSM_pt[i][j].x) * P_last_tmp[j];
                    y_update += (BM_pt[i][j].y - DSM_pt[i][j].y) * P_last_tmp[j];
                    z_update += (BM_pt[i][j].z - DSM_pt[i][j].z) * P_last_tmp[j];
                }
                else
                {
                    pt_total_P += P_last[i][j];
                    x_update += (BM_pt[i][j].x - DSM_pt[i][j].x) * P_last[i][j];
                    y_update += (BM_pt[i][j].y - DSM_pt[i][j].y) * P_last[i][j];
                    z_update += (BM_pt[i][j].z - DSM_pt[i][j].z) * P_last[i][j];
                }
            }
            P_last.push_back(P_last_tmp);
        }
        x_update /= pt_total_P;
        y_update /= pt_total_P;
        z_update /= pt_total_P;
        if (abs(x_update - Model_Para[0]) + abs(y_update - Model_Para[1]) + abs(z_update - Model_Para[2]) < 1e-3)
        {
            Point3d Model_Para_tmp;
            Model_Para_tmp.x = Model_Para[0];
            Model_Para_tmp.y = Model_Para[1];
            Model_Para_tmp.z = Model_Para[2];
            Model_Translation_Parameter_WEIGHT_ITER = Model_Para_tmp;

            break;
        }
        else
        {
            Model_Para[0] = x_update;
            Model_Para[1] = y_update;
            Model_Para[2] = z_update;
        }
    }
}

void Weight_Iteration_Point_Elevation(vector <vector<Point3d>> BM_pt, vector <vector<Point3d>> DSM_pt, double Init_Para_x, double Init_Para_y, double Init_Para_z, double sigma_bef, Point3d& Model_Translation_Parameter_WEIGHT_ITER, vector<vector<double>> finalConfidenceMatrix, string dir_path)
{
    vector<string> tif_dem;
    string BM_dem_path = dir_path + "\\" + "BASEMAP" + "\\" + "SRTM";
    GetAllFiles_tif(BM_dem_path, tif_dem);

    GDALAllRegister();
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名

    // 读取SRTM数据
    GDALDataset* poDataset = (GDALDataset*)GDALOpen((BM_dem_path + "\\" + tif_dem[0]).c_str(), GA_ReadOnly);
    if (poDataset == NULL) {
        cout << "无法读取SRTM数据！" << endl;
        return;
    }

    int nXSize = poDataset->GetRasterXSize();
    int nYSize = poDataset->GetRasterYSize();
    double adfGeoTransform[6];
    poDataset->GetGeoTransform(adfGeoTransform);
    double cellSizeX = adfGeoTransform[1]; // 一个格子的宽度
    double cellSizeY = -adfGeoTransform[5]; // 一个格子的高度（通常为负值）

    // 将最终置信度应用到BM_pt
    // // 初始化 Elevation_Weight 矩阵，使其与 BM_pt 一样大小，并填充为0
    vector<vector<double>> Elevation_Weight;
    Elevation_Weight.resize(BM_pt.size());
    for (size_t i = 0; i < BM_pt.size(); ++i) {
        Elevation_Weight[i].resize(BM_pt[i].size(), 0.0);
    }
    // 在应用置信度到 BM_pt 之前加上断言
    //assert(Elevation_Weight.size() == BM_pt.size());
    for (size_t i = 0; i < BM_pt.size(); ++i) {
        for (size_t j = 0; j < BM_pt[i].size(); ++j) {
            int xIndex = static_cast<int>((BM_pt[i][j].x - adfGeoTransform[0]) / cellSizeX);
            int yIndex = static_cast<int>((BM_pt[i][j].y - adfGeoTransform[3]) / -cellSizeY);
            if (finalConfidenceMatrix[yIndex][xIndex] > 0.8)
            {
                Elevation_Weight[i][j] = finalConfidenceMatrix[yIndex][xIndex];
            }
        }
    }

    FILE* f_ele_weight;
    f_ele_weight = fopen((BM_dem_path + "\\elevation_weight.txt").c_str(), "w");
    for (int fi = 0; fi < finalConfidenceMatrix.size(); fi++)
    {
        for (int fj = 0; fj < finalConfidenceMatrix[fi].size(); fj++)
        {
            fprintf(f_ele_weight, "%lf ", finalConfidenceMatrix[fi][fj]);
        }
        fprintf(f_ele_weight, "\n");
    }
    fclose(f_ele_weight);

    GDALAllRegister(); // 初始化GDAL


    // 创建高程底图的elevation weight图像
//// 打开原始影像
//    GDALDataset* poSrcDS = (GDALDataset*)GDALOpen((BM_dem_path + "\\" + tif_dem[0]).c_str(), GA_ReadOnly);
//    if (poSrcDS == nullptr) {
//        std::cerr << "无法打开原始影像文件" << std::endl;
//    }
//
//    // 获取原始影像的地理信息和数据类型
//    auto poSrcGeoTransform = new double[6];
//    poSrcDS->GetGeoTransform(poSrcGeoTransform);
//    const char* pszSrcWKT = poSrcDS->GetProjectionRef();
//    int xSize = poSrcDS->GetRasterXSize();
//    int ySize = poSrcDS->GetRasterYSize();
//    GDALDataType eType = GDT_Float64;
//
//    // 创建新影像
//    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
//    GDALDataset* poDstDS = poDriver->Create((BM_dem_path + "\\Elevation_Weight.tif").c_str(), xSize, ySize, 1, eType, NULL);
//    poDstDS->SetGeoTransform(poSrcGeoTransform);
//    poDstDS->SetProjection(pszSrcWKT);
//
//    // 将array的值赋给新影像
//    for (int i = 0; i < ySize; ++i) {
//        poDstDS->GetRasterBand(1)->RasterIO(GF_Write, 0, i, xSize, 1, finalConfidenceMatrix[i].data(), xSize, 1, GDT_Float64, 0, 0);
//    }
//
//    // 清理和关闭数据集
//    GDALClose(poDstDS);
//    GDALClose(poSrcDS);
//    delete[] poSrcGeoTransform;




    double z_upgrate = 0;
    double total_elevation_weight = 0;
    for (int i = 0; i < BM_pt.size(); ++i) {
        for (int j = 0; j < BM_pt[i].size(); ++j) {
            z_upgrate += Elevation_Weight[i][j] * (BM_pt[i][j].z - DSM_pt[i][j].z);
            total_elevation_weight += Elevation_Weight[i][j];
        }
    }
    z_upgrate /= total_elevation_weight;
    Init_Para_z = z_upgrate;

    double Model_Para[3];
    Model_Para[0] = Init_Para_x;
    Model_Para[1] = Init_Para_y;
    Model_Para[2] = Init_Para_z;

    // double V_sum;
    int match_num = BM_pt.size();//外层控制匹配对数目
    int iter_time = 0;
    vector<vector <double>> P_last;
    while (1)
    {
        iter_time++;
        double pt_total_P = 0;
        double pt_total_P_z = 0;
        double x_update = 0;
        double y_update = 0;
        double z_update = 0;

        for (int i = 0; i < match_num; i++)//外层控制，匹配对数目
        {
            int match_pt_num = BM_pt[i].size();//内层控制，各个匹配对中点的数目
            vector <double> P_last_tmp;
            for (int j = 0; j < match_pt_num; j++)
            {
                //printf("BM_pt i %d j %d\n", BM_pt.size(), BM_pt[i].size());
                //printf("Elevation_Weight i %d j %d\n", Elevation_Weight.size(),Elevation_Weight[i].size());
                //计算当前模型下的点校正后坐标，与底图控制点坐标之间的差距即为V_dist
                double x_rect = DSM_pt[i][j].x + Model_Para[0];
                double y_rect = DSM_pt[i][j].y + Model_Para[1];
                double z_rect = DSM_pt[i][j].z + Model_Para[2];
                double dx = abs(BM_pt[i][j].x - x_rect);
                double dy = abs(BM_pt[i][j].y - y_rect);
                double dz = abs(BM_pt[i][j].z - z_rect);
                double V_dist = sqrt(dx * dx + dy * dy + dz * dz);
                //IGG3权函数，k0取2.5，k1取3.5
                double k0 = K_0, k1 = K_1;
                if (V_dist / sigma_bef < k0) // 可信段
                {
                    if (iter_time == 1)
                    {
                        P_last_tmp.push_back(1.0);
                    }
                }
                if (V_dist / sigma_bef >= k0 && V_dist / sigma_bef < k1) // 可疑段
                {
                    double d = (k1 - (V_dist / sigma_bef)) / (k1 - k0);
                    double w = d * d * k0 / (V_dist / sigma_bef);
                    if (iter_time == 1)
                    {
                        P_last_tmp.push_back(w);
                    }
                    else
                    {
                        P_last[i][j] = P_last[i][j] * w;//要记录上一次的权重
                    }
                }
                if (V_dist / sigma_bef >= k1) // 粗差段
                {
                    if (iter_time == 1)
                    {
                        P_last_tmp.push_back(0);
                    }
                    else
                    {
                        P_last[i][j] = 0;
                    }
                }
                //计算加权平均值
                if (iter_time == 1)
                {
                    pt_total_P += P_last_tmp[j];
                    pt_total_P_z += P_last_tmp[j] * Elevation_Weight[i][j];
                    x_update += (BM_pt[i][j].x - DSM_pt[i][j].x) * P_last_tmp[j];
                    y_update += (BM_pt[i][j].y - DSM_pt[i][j].y) * P_last_tmp[j];
                    z_update += (BM_pt[i][j].z - DSM_pt[i][j].z) * P_last_tmp[j] * Elevation_Weight[i][j];
                }
                else
                {
                    pt_total_P += P_last[i][j];
                    pt_total_P_z += P_last[i][j] * Elevation_Weight[i][j];
                    x_update += (BM_pt[i][j].x - DSM_pt[i][j].x) * P_last[i][j];
                    y_update += (BM_pt[i][j].y - DSM_pt[i][j].y) * P_last[i][j];
                    z_update += (BM_pt[i][j].z - DSM_pt[i][j].z) * P_last[i][j] * Elevation_Weight[i][j];
                }
            }
            P_last.push_back(P_last_tmp);
        }
        x_update /= pt_total_P;
        y_update /= pt_total_P;
        z_update /= pt_total_P_z;
        if (abs(x_update - Model_Para[0]) + abs(y_update - Model_Para[1]) + abs(z_update - Model_Para[2]) < 1e-3)
        {
            Point3d Model_Para_tmp;
            Model_Para_tmp.x = Model_Para[0];
            Model_Para_tmp.y = Model_Para[1];
            Model_Para_tmp.z = Model_Para[2];
            Model_Translation_Parameter_WEIGHT_ITER = Model_Para_tmp;

            break;
        }
        else
        {
            Model_Para[0] = x_update;
            Model_Para[1] = y_update;
            Model_Para[2] = z_update;
        }
    }
}

void Weight_Iteration_Point_old(vector <vector<Point3d>> BM_pt, vector <vector<Point3d>> DSM_pt, double Init_Para_x, double Init_Para_y, double Init_Para_z, double sigma_bef, Point3d& Model_Translation_Parameter_WEIGHT_ITER)
{
    double Model_Para[3];
    Model_Para[0] = Init_Para_x;
    Model_Para[1] = Init_Para_y;
    Model_Para[2] = Init_Para_z;

    // double V_sum;
    int match_num = BM_pt.size();//外层控制匹配对数目
    int iter_time = 0;
    vector <double> P_last;

    while (1)
    {
        iter_time++;
        double pt_total_P = 0;
        double x_update = 0;
        double y_update = 0;
        double z_update = 0;

        for (int i = 0; i < match_num; i++)//外层控制，匹配对数目
        {
            int match_pt_num = BM_pt[i].size();//内层控制，各个匹配对中点的数目
            vector <double> P_tmp;
            for (int j = 0; j < match_pt_num; j++)
            {
                //计算当前模型下的点校正后坐标，与底图控制点坐标之间的差距即为V_dist
                double x_rect = DSM_pt[i][j].x + Model_Para[0];
                double y_rect = DSM_pt[i][j].y + Model_Para[1];
                double z_rect = DSM_pt[i][j].z + Model_Para[2];
                double dx = abs(BM_pt[i][j].x - x_rect);
                double dy = abs(BM_pt[i][j].y - y_rect);
                double dz = abs(BM_pt[i][j].z - z_rect);
                double V_dist = sqrt(dx * dx + dy * dy + dz * dz);
                //IGG3权函数，k0取2.5，k1取3.5
                double k0 = K_0, k1 = K_1;
                if (V_dist / sigma_bef < k0) // 可信段
                {
                    if (iter_time == 1)
                    {
                        P_last.push_back(1.0);
                    }
                    else
                    {
                        P_tmp.push_back(P_last[j]);
                    }
                }
                if (V_dist / sigma_bef >= k0 && V_dist / sigma_bef < k1) // 可疑段
                {
                    double d = (k1 - (V_dist / sigma_bef)) / (k1 - k0);
                    double w = d * d * k0 / (V_dist / sigma_bef);
                    if (iter_time == 1)
                    {
                        P_last.push_back(w);
                    }
                    else
                    {
                        P_tmp.push_back(P_last[j] * w);//要记录上一次的权重
                    }
                }
                if (V_dist / sigma_bef >= k1) // 粗差段
                {
                    if (iter_time == 1)
                    {
                        P_last.push_back(0);
                    }
                    else
                    {
                        P_tmp.push_back(0);
                    }
                }
                //计算加权平均值
                if (iter_time == 1)
                {
                    pt_total_P += P_last[j];
                    x_update += (BM_pt[i][j].x - DSM_pt[i][j].x) * P_last[j];
                    y_update += (BM_pt[i][j].y - DSM_pt[i][j].y) * P_last[j];
                    z_update += (BM_pt[i][j].z - DSM_pt[i][j].z) * P_last[j];
                }
                else
                {
                    pt_total_P += P_tmp[j];
                    x_update += (BM_pt[i][j].x - DSM_pt[i][j].x) * P_tmp[j];
                    y_update += (BM_pt[i][j].y - DSM_pt[i][j].y) * P_tmp[j];
                    z_update += (BM_pt[i][j].z - DSM_pt[i][j].z) * P_tmp[j];
                }
            }
        }
        x_update /= pt_total_P;
        y_update /= pt_total_P;
        z_update /= pt_total_P;
        if (abs(x_update - Model_Para[0]) + abs(y_update - Model_Para[1]) + abs(z_update - Model_Para[2]) < 1e-3)
        {
            Point3d Model_Para_tmp;
            Model_Para_tmp.x = Model_Para[0];
            Model_Para_tmp.y = Model_Para[1];
            Model_Para_tmp.z = Model_Para[2];
            Model_Translation_Parameter_WEIGHT_ITER = Model_Para_tmp;

            break;
        }
        else
        {
            Model_Para[0] = x_update;
            Model_Para[1] = y_update;
            Model_Para[2] = z_update;
        }
    }
}

void DSM_Rectified(const char* pszSrcFile, const char* pszDstFile, const char* pszFormat, double x, double y, double z)
{
    GDALAllRegister();
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名

    GDALDataset* pSrcDS = (GDALDataset*)GDALOpen(pszSrcFile, GA_ReadOnly);
    GDALDataType eDT = pSrcDS->GetRasterBand(1)->GetRasterDataType();

    int iBandCount = pSrcDS->GetRasterCount();

    //根据裁切范围确定裁切后的图像宽高)
    int iDstWidth = pSrcDS->GetRasterXSize();
    int iDstHeight = pSrcDS->GetRasterYSize();

    double adfGeoTransform[6] = { 0 };
    pSrcDS->GetGeoTransform(adfGeoTransform);

    //计算裁切后的图像的左上角坐标
    adfGeoTransform[0] = adfGeoTransform[0] + x;//应该是加，2022/11/15 21点，经过慎重思考和画图，认为，x和y是左上角原点的变化，x为横轴，y为向上的纵轴，做出以下修改，认为是减
    adfGeoTransform[3] = adfGeoTransform[3] + y;

    //创建输出文件并设置空间参考和坐标信息
    GDALDriver* poDriver = (GDALDriver*)GDALGetDriverByName(pszFormat);
    GDALDataset* pDstDS = poDriver->Create(pszDstFile, iDstWidth, iDstHeight, iBandCount, eDT, NULL);
    pDstDS->SetGeoTransform(adfGeoTransform);
    //pDstDS->SetProjection(pSrcDS->GetProjectionRef());

    int* pBandMap = new int[iBandCount];
    for (int i = 0; i < iBandCount; i++)
    {
        pBandMap[i] = i + 1;
    }

    if (eDT == GDT_Byte)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        unsigned int* pDataBuff = new unsigned int[iDstWidth * iDstHeight * iBandCount];
        pSrcDS->RasterIO(GF_Read, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);
        for (int band_num = 1; band_num <= iBandCount; band_num++)
        {
            for (int h = 0; h < iDstHeight; h++)
            {
                for (int w = 0; w < iDstWidth; w++)
                {
                    //if (pDataBuff[(w + h * iDstWidth) * band_num] > 53)
                    //	int comeon0 = 0;
                    pDataBuff[(w + h * iDstWidth) * band_num] = pDataBuff[(w + h * iDstWidth) * band_num] + z;
                    //if (pDataBuff[(w + h * iDstWidth) * band_num] < -1000)
                    //	int comeon = 0;
                }
            }
        }

        pDstDS->RasterIO(GF_Write, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        RELEASE(pDataBuff);
    }
    else if (eDT == GDT_UInt16)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        unsigned short* pDataBuff = new unsigned short[iDstWidth * iDstHeight * iBandCount];
        pSrcDS->RasterIO(GF_Read, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);
        for (int band_num = 1; band_num <= iBandCount; band_num++)
        {
            for (int h = 0; h < iDstHeight; h++)
            {
                for (int w = 0; w < iDstWidth; w++)
                {
                    if (pDataBuff[(w + h * iDstWidth) * band_num] > 53)
                        int comeon0 = 0;
                    pDataBuff[(w + h * iDstWidth) * band_num] = pDataBuff[(w + h * iDstWidth) * band_num] + z;
                    if (pDataBuff[(w + h * iDstWidth) * band_num] < -1000)
                        int comeon = 0;
                }
            }
        }

        pDstDS->RasterIO(GF_Write, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        RELEASE(pDataBuff);
    }
    else if (eDT == GDT_Float32)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        float* pDataBuff = new float[iDstWidth * iDstHeight * iBandCount];
        pSrcDS->RasterIO(GF_Read, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        //校正DSM的高程
        for (int band_num = 1; band_num <= iBandCount; band_num++)
        {
            for (int h = 0; h < iDstHeight; h++)
            {
                for (int w = 0; w < iDstWidth; w++)
                {
                    pDataBuff[(w + h * iDstWidth) * band_num] = pDataBuff[(w + h * iDstWidth) * band_num] + z;
                }
            }
        }
        pDstDS->RasterIO(GF_Write, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        RELEASE(pDataBuff);
    }

    RELEASE(pBandMap);

    GDALClose((GDALDatasetH)pSrcDS);
    GDALClose((GDALDatasetH)pDstDS);
    GDALDestroyDriverManager();
    return;
}


void DOM_Rectified(const char* pszSrcFile, const char* pszDstFile, const char* pszFormat, double x, double y)
{
    GDALAllRegister();
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名

    GDALDataset* pSrcDS = (GDALDataset*)GDALOpen(pszSrcFile, GA_ReadOnly);
    GDALDataType eDT = pSrcDS->GetRasterBand(1)->GetRasterDataType();

    int iBandCount = pSrcDS->GetRasterCount();

    //根据裁切范围确定裁切后的图像宽高)
    int iDstWidth = pSrcDS->GetRasterXSize();
    int iDstHeight = pSrcDS->GetRasterYSize();

    double adfGeoTransform[6] = { 0 };
    pSrcDS->GetGeoTransform(adfGeoTransform);

    //计算裁切后的图像的左上角坐标
    adfGeoTransform[0] = adfGeoTransform[0] + x;
    adfGeoTransform[3] = adfGeoTransform[3] + y;

    //创建输出文件并设置空间参考和坐标信息
    GDALDriver* poDriver = (GDALDriver*)GDALGetDriverByName(pszFormat);
    GDALDataset* pDstDS = poDriver->Create(pszDstFile, iDstWidth, iDstHeight, iBandCount, eDT, NULL);
    pDstDS->SetGeoTransform(adfGeoTransform);
    //pDstDS->SetProjection(pSrcDS->GetProjectionRef());

    int* pBandMap = new int[iBandCount];
    for (int i = 0; i < iBandCount; i++)
    {
        pBandMap[i] = i + 1;
    }

    if (eDT == GDT_Byte)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        unsigned int* pDataBuff = new unsigned int[iDstWidth * iDstHeight * iBandCount];
        pSrcDS->RasterIO(GF_Read, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);
        pDstDS->RasterIO(GF_Write, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        RELEASE(pDataBuff);
    }
    else if (eDT == GDT_UInt16)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        unsigned short* pDataBuff = new unsigned short[iDstWidth * iDstHeight * iBandCount];
        pSrcDS->RasterIO(GF_Read, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);
        pDstDS->RasterIO(GF_Write, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        RELEASE(pDataBuff);
    }
    else if (eDT == GDT_Float32)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        float* pDataBuff = new float[iDstWidth * iDstHeight * iBandCount];
        pSrcDS->RasterIO(GF_Read, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);
        pDstDS->RasterIO(GF_Write, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        RELEASE(pDataBuff);
    }

    RELEASE(pBandMap);

    GDALClose((GDALDatasetH)pSrcDS);
    GDALClose((GDALDatasetH)pDstDS);
    GDALDestroyDriverManager();
    return;
}


void Basemap_Cut(string Input_Basemap_Dir, string Output_Dir)
{
    vector <string> Input_ARC;
    GetAllFiles_tif(Input_Basemap_Dir + "\\ARC_BING_MB", Input_ARC);
    vector <string> Input_GE;
    GetAllFiles_tif(Input_Basemap_Dir + "\\GE", Input_GE);
    vector <string> Output_Areas_Dir;
    GetFolder(Output_Dir, Output_Areas_Dir);
    for (int i = 0; i < Output_Areas_Dir.size(); i++)
    {
        Output_Areas_Dir[i] = Output_Dir + "\\" + Output_Areas_Dir[i];
        if (0 != _access((Output_Areas_Dir[i] + "\\BASEMAP").c_str(), 0))
        {
            int result = _mkdir((Output_Areas_Dir[i] + "\\BASEMAP").c_str());
            if (result != 0) {
                switch (errno) {
                case ENOENT:
                    std::cerr << "Path was not found." << std::endl;
                    break;
                default:
                    std::cerr << "Unknown error while creating directory." << std::endl;
                    break;
                }
            }
        }
        if (0 != _access((Output_Areas_Dir[i] + "\\BASEMAP\\ARC_BING_MB").c_str(), 0))
        {
            int result = _mkdir((Output_Areas_Dir[i] + "\\BASEMAP\\ARC_BING_MB").c_str());
            if (result != 0) {
                switch (errno) {
                case ENOENT:
                    std::cerr << "Path was not found." << std::endl;
                    break;
                default:
                    std::cerr << "Unknown error while creating directory." << std::endl;
                    break;
                }
            }
        }
        if (0 != _access((Output_Areas_Dir[i] + "\\BASEMAP\\GE").c_str(), 0))
        {
            int result = _mkdir((Output_Areas_Dir[i] + "\\BASEMAP\\GE").c_str());
            if (result != 0) {
                switch (errno) {
                case ENOENT:
                    std::cerr << "Path was not found." << std::endl;
                    break;
                default:
                    std::cerr << "Unknown error while creating directory." << std::endl;
                    break;
                }
            }
        }
        if (0 != _access((Output_Areas_Dir[i] + "\\BASEMAP\\SRTM").c_str(), 0))
        {
            int result = _mkdir((Output_Areas_Dir[i] + "\\BASEMAP\\SRTM").c_str());
            if (result != 0) {
                switch (errno) {
                case ENOENT:
                    std::cerr << "Path was not found." << std::endl;
                    break;
                default:
                    std::cerr << "Unknown error while creating directory." << std::endl;
                    break;
                }
            }
        }
        //先放SRTM数据,orthophoto.tif
        DOM_Rectified((Input_Basemap_Dir + "\\SRTM\\SRTM.tif").c_str(), (Output_Areas_Dir[i] + "\\BASEMAP\\SRTM\\SRTM.tif").c_str(), "GTiff", 0, 0);

        //裁剪该区域的ARC_BING_MB
        for (int num_ARC = 0; num_ARC < Input_ARC.size(); num_ARC++)
        {
            GDALAllRegister();//注册所有的驱动
            CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名
            //1、加载tif数据
            vector <string> Area_DOM_tif;
            vector <string> Area_DOM_tfw;
            GetAllFiles_tif(Output_Areas_Dir[i] + "\\DOM", Area_DOM_tif);
            GetAllFiles_tfw(Output_Areas_Dir[i] + "\\DOM", Area_DOM_tfw);

            string tif_path_name = Output_Areas_Dir[i] + "\\DOM\\" + Area_DOM_tif[0];
            string tfw_path_name = Output_Areas_Dir[i] + "\\DOM\\" + Area_DOM_tfw[0];
            if (num_ARC == 0)
            {
                DOM_Rectified(tif_path_name.c_str(), (Output_Areas_Dir[i] + "\\BASEMAP\\ARC_BING_MB\\" + Area_DOM_tif[0]).c_str(), "GTiff", 0, 0);
                GDALAllRegister();
            }

            GDALDataset* poDataset = (GDALDataset*)GDALOpen(tif_path_name.c_str(), GA_Update);//GA_Update和GA_ReadOnly两种模式
            if (poDataset == NULL)
                std::cout << tif_path_name <<"指定的文件不能打开!" << std::endl;
            //获取图像的尺寸
            int nImgSizeX = poDataset->GetRasterXSize();
            int nImgSizeY = poDataset->GetRasterYSize();

            double dom_trans[6] = { 0 };

            FILE* p_dom;
            p_dom = fopen(tfw_path_name.c_str(), "r");
            fscanf(p_dom, "%lf", &dom_trans[1]);
            fscanf(p_dom, "%lf", &dom_trans[2]);
            fscanf(p_dom, "%lf", &dom_trans[4]);
            fscanf(p_dom, "%lf", &dom_trans[5]);
            fscanf(p_dom, "%lf", &dom_trans[0]);
            fscanf(p_dom, "%lf", &dom_trans[3]);
            poDataset->SetGeoTransform(dom_trans);
            fclose(p_dom);

            poDataset->SetGeoTransform(dom_trans);

            double DOM_width = dom_trans[1] * nImgSizeX;
            double DOM_height = -dom_trans[5] * nImgSizeY;

            GDALDataset* poDataset_BM = (GDALDataset*)GDALOpen((Input_Basemap_Dir + "\\ARC_BING_MB\\" + Input_ARC[num_ARC]).c_str(), GA_ReadOnly);//GA_Update和GA_ReadOnly两种模式
            if (poDataset_BM == NULL)
                std::cout << Input_Basemap_Dir + "\\ARC_BING_MB\\" + Input_ARC[num_ARC] << "指定的文件不能打开!" << std::endl;

            double BM_trans[6] = { 0 };
            poDataset_BM->GetGeoTransform(BM_trans);

            //求出起始行列号，左上，仅考虑无旋转的情况
            double StartX = (dom_trans[0] - BM_trans[0]) / BM_trans[1];
            double StartY = (dom_trans[3] - BM_trans[3]) / BM_trans[5];

            if (StartX < 0) StartX = 0;
            if (StartY < 0) StartY = 0;
            if (StartX > poDataset_BM->GetRasterXSize() - 1)printf("%s区域不在底图x范围内", Output_Areas_Dir[i].c_str());
            if (StartY > poDataset_BM->GetRasterYSize() - 1)printf("%s区域不在底图y范围内", Output_Areas_Dir[i].c_str());

            int dx = DOM_width / BM_trans[1];
            int dy = -DOM_height / BM_trans[5];
            if (StartX + dx > poDataset_BM->GetRasterXSize() - 1) dx = poDataset_BM->GetRasterXSize() - 1 - StartX;
            if (StartY + dy > poDataset_BM->GetRasterYSize() - 1) dy = poDataset_BM->GetRasterYSize() - 1 - StartY;

            ImageCut((Input_Basemap_Dir + "\\ARC_BING_MB\\" + Input_ARC[num_ARC]).c_str(), (Output_Areas_Dir[i] + "\\BASEMAP\\ARC_BING_MB\\" + Input_ARC[num_ARC]).c_str(), StartX, StartY, dx, dy, "GTiff");
            poDataset = NULL;
            poDataset_BM = NULL;
        }
        //裁剪该区域的GE
        for (int num_GE = 0; num_GE < Input_GE.size(); num_GE++)
        {
            GDALAllRegister();//注册所有的驱动
            CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名
            //1、加载tif数据
            vector <string> Area_DOM_tif;
            vector <string> Area_DOM_tfw;
            GetAllFiles_tif(Output_Areas_Dir[i] + "\\DOM", Area_DOM_tif);
            GetAllFiles_tfw(Output_Areas_Dir[i] + "\\DOM", Area_DOM_tfw);

            string tif_path_name = Output_Areas_Dir[i] + "\\DOM\\" + Area_DOM_tif[0];
            string tfw_path_name = Output_Areas_Dir[i] + "\\DOM\\" + Area_DOM_tfw[0];

            if (num_GE == 0)
            {
                DOM_Rectified(tif_path_name.c_str(), (Output_Areas_Dir[i] + "\\BASEMAP\\GE\\" + Area_DOM_tif[0]).c_str(), "GTiff", 0, 0);
                GDALAllRegister();
            }

            GDALDataset* poDataset = (GDALDataset*)GDALOpen(tif_path_name.c_str(), GA_Update);//GA_Update和GA_ReadOnly两种模式
            if (poDataset == NULL)
                std::cout << "指定的文件不能打开!" << std::endl;
            //获取图像的尺寸
            int nImgSizeX = poDataset->GetRasterXSize();
            int nImgSizeY = poDataset->GetRasterYSize();

            double dom_trans[6] = { 0 };

            FILE* p_dom;
            p_dom = fopen(tfw_path_name.c_str(), "r");
            fscanf(p_dom, "%lf", &dom_trans[1]);
            fscanf(p_dom, "%lf", &dom_trans[2]);
            fscanf(p_dom, "%lf", &dom_trans[4]);
            fscanf(p_dom, "%lf", &dom_trans[5]);
            fscanf(p_dom, "%lf", &dom_trans[0]);
            fscanf(p_dom, "%lf", &dom_trans[3]);
            poDataset->SetGeoTransform(dom_trans);
            fclose(p_dom);

            poDataset->SetGeoTransform(dom_trans);

            double DOM_width = dom_trans[1] * nImgSizeX;
            double DOM_height = -dom_trans[5] * nImgSizeY;

            GDALDataset* poDataset_BM = (GDALDataset*)GDALOpen((Input_Basemap_Dir + "\\GE\\" + Input_GE[num_GE]).c_str(), GA_ReadOnly);//GA_Update和GA_ReadOnly两种模式
            if (poDataset_BM == NULL)
                std::cout << Input_Basemap_Dir + "\\GE\\" + Input_GE[num_GE] << "指定的文件不能打开!" << std::endl;

            double BM_trans[6] = { 0 };
            poDataset_BM->GetGeoTransform(BM_trans);

            //求出起始行列号，左上，仅考虑无旋转的情况
            double StartX = (dom_trans[0] - BM_trans[0]) / BM_trans[1];
            double StartY = (dom_trans[3] - BM_trans[3]) / BM_trans[5];

            if (StartX < 0) StartX = 0;
            if (StartY < 0) StartY = 0;
            if (StartX > poDataset_BM->GetRasterXSize() - 1)printf("%s区域不在底图x范围内", Output_Areas_Dir[i].c_str());
            if (StartY > poDataset_BM->GetRasterYSize() - 1)printf("%s区域不在底图y范围内", Output_Areas_Dir[i].c_str());

            int dx = DOM_width / BM_trans[1];
            int dy = -DOM_height / BM_trans[5];
            if (StartX + dx > poDataset_BM->GetRasterXSize() - 1) dx = poDataset_BM->GetRasterXSize() - 1 - StartX;
            if (StartY + dy > poDataset_BM->GetRasterYSize() - 1) dy = poDataset_BM->GetRasterYSize() - 1 - StartY;


            ImageCut((Input_Basemap_Dir + "\\GE\\" + Input_GE[num_GE]).c_str(), (Output_Areas_Dir[i] + "\\BASEMAP\\GE\\" + Input_GE[num_GE]).c_str(), StartX, StartY, dx, dy, "GTiff");
            poDataset = NULL;
            poDataset_BM = NULL;

        }
    }
}

void Precision_Evaluation(const char* TruthFile, const char* RectFile, string DstDir, string Area, bool is_Write_Heatmap)
{
    //精度评价，输入两张影像的地址，输出误差值的矩阵txt，误差绝对值的矩阵txt
    GDALAllRegister();
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名

    GDALDataset* pTruthDS = (GDALDataset*)GDALOpen(TruthFile, GA_ReadOnly);
    GDALDataType eDT = pTruthDS->GetRasterBand(1)->GetRasterDataType();

    //根据裁切范围确定裁切后的图像宽高)
    int iDstWidth = pTruthDS->GetRasterXSize();
    int iDstHeight = pTruthDS->GetRasterYSize();

    double truGeoTransform[6] = { 0 };
    pTruthDS->GetGeoTransform(truGeoTransform);

    int iBandCount = pTruthDS->GetRasterCount();

    int* pBandMap = new int[iBandCount];
    for (int i = 0; i < iBandCount; i++)
    {
        pBandMap[i] = i + 1;
    }

    GDALDataset* pRectDS = (GDALDataset*)GDALOpen(RectFile, GA_ReadOnly);
    GDALDataType eDT1 = pRectDS->GetRasterBand(1)->GetRasterDataType();
    //根据裁切范围确定裁切后的图像宽高)
    int iRectWidth = pRectDS->GetRasterXSize();
    int iRectHeight = pRectDS->GetRasterYSize();

    double RectGeoTransform[6] = { 0 };
    pRectDS->GetGeoTransform(RectGeoTransform);

    int iBandCountRect = pRectDS->GetRasterCount();

    int* pBandMapRect = new int[iBandCountRect];
    for (int i = 0; i < iBandCountRect; i++)
    {
        pBandMapRect[i] = i + 1;
    }
    //读取影像
    if (eDT == GDT_Float32 && eDT1 == GDT_Float32)
    {
        //申请所有数据时所需要的缓存
        float* pDataBuff = new float[iDstWidth * iDstHeight * iBandCount];
        pTruthDS->RasterIO(GF_Read, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        float* pDataBuffRect = new float[iRectWidth * iRectHeight * iBandCountRect];
        pRectDS->RasterIO(GF_Read, 0, 0, iRectWidth, iRectHeight, pDataBuffRect, iRectWidth, iRectHeight, eDT1, iBandCountRect, pBandMapRect, 0, 0, 0);

        //相减,注意以真值的图像范围为矩阵的大小，先计算平均值，注意粗差（abs > 20）及NaN值记为99999，且n--
        float* Error_Matrix = new float[iDstWidth * iDstHeight];
        float* Error_Matrix_Abs = new float[iDstWidth * iDstHeight];

        double mean = 0, sigma = 0;
        int num = 0;
        //计算图像地理坐标,若图像中某一点的行数和列数分别为：row, column,则该点的地理坐标为：
        //由于真值一般小于该区域，故将该区域范围作为误差矩阵范围
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {

                Point3d Geo_pt;
                Geo_pt.x = truGeoTransform[0] + (wid_i + 0.5) * truGeoTransform[1] + hgt_i * truGeoTransform[2];
                Geo_pt.y = truGeoTransform[3] + wid_i * truGeoTransform[4] + (hgt_i + 0.5) * truGeoTransform[5];

                //计算行列号
                double dTemp = RectGeoTransform[2] * RectGeoTransform[4] - RectGeoTransform[1] * RectGeoTransform[5];
                double dRow = (RectGeoTransform[4] * (Geo_pt.x - RectGeoTransform[0]) - RectGeoTransform[1] * (Geo_pt.y - RectGeoTransform[3])) / dTemp;
                double dCol = (RectGeoTransform[2] * (Geo_pt.y - RectGeoTransform[3]) - RectGeoTransform[5] * (Geo_pt.x - RectGeoTransform[0])) / dTemp;
                int dx = int(dCol);
                int dy = int(dRow);
                if (dy < 0 || dx < 0 || dx >= iRectWidth || dy >= iRectHeight)
                    Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                else
                {
                    float tmp_Err = pDataBuff[hgt_i * iDstWidth + wid_i] - pDataBuffRect[dy * iRectWidth + dx];
                    if (!isnan(tmp_Err) && abs(tmp_Err) < 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = tmp_Err;
                        mean += tmp_Err;
                        num++;
                    }
                    else if (abs(tmp_Err) > 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                    else if (isnan(tmp_Err))
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }

            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
                }
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);
        //去除异常值，将3sigma外的异常值去掉，取绝对值得到误差绝对值的矩阵
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] > mean + ABNORMAL_VALUE_THRESHOLD * sigma || Error_Matrix[hgt_i * iDstWidth + wid_i] < mean - ABNORMAL_VALUE_THRESHOLD * sigma)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                }
                else
                {
                    Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                }
            }
        }
        mean = 0, sigma = 0, num = 0;

        //输出到同一个指定文件夹中，空值写为NaN
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    mean += Error_Matrix_Abs[hgt_i * iDstWidth + wid_i];
                    num++;
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    sigma += (Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] - mean);
                }
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);

        //输出txt文件
        string Type_Str = string(RectFile).substr(string(RectFile).length() - 6, 2);
        string Print_Str = Area + "_" + Type_Str;
        printf("%s %lf %lf\n", Print_Str.c_str(), mean, sigma);

        if (is_Write_Heatmap == true)
        {
            string opt_txt1 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix.txt";
            FILE* p1;
            p1 = fopen(opt_txt1.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {

                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p1, "%lf ", Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p1, "NaN ");
                    }
                }
                fprintf(p1, "\n");
            }
            fclose(p1);

            string opt_txt2 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix_Abs.txt";
            FILE* p2;
            p2 = fopen(opt_txt2.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {

                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p2, "%lf ", Error_Matrix_Abs[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p2, "NaN ");
                    }
                }
                fprintf(p2, "\n");
            }
            fclose(p2);
        }

        RELEASE(Error_Matrix);
        RELEASE(Error_Matrix_Abs);
        RELEASE(pDataBuff);
        RELEASE(pDataBuffRect);
    }
    else if (eDT == GDT_Byte && eDT1 == GDT_Float32)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        unsigned int* pDataBuff = new unsigned int[iDstWidth * iDstHeight * iBandCount];
        pTruthDS->RasterIO(GF_Read, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        float* pDataBuffRect = new float[iRectWidth * iRectHeight * iBandCountRect];
        pRectDS->RasterIO(GF_Read, 0, 0, iRectWidth, iRectHeight, pDataBuffRect, iRectWidth, iRectHeight, eDT1, iBandCountRect, pBandMapRect, 0, 0, 0);

        //相减,注意以真值的图像范围为矩阵的大小，先计算平均值，注意粗差（abs > 20）及NaN值记为99999，且n--

        float* Error_Matrix = new float[iDstWidth * iDstHeight];
        float* Error_Matrix_Abs = new float[iDstWidth * iDstHeight];

        double mean = 0, sigma = 0;
        int num = 0;
        //计算图像地理坐标,若图像中某一点的行数和列数分别为：row, column,则该点的地理坐标为：
        for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
        {
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                Point3d Geo_pt;
                Geo_pt.x = truGeoTransform[0] + (wid_i + 0.5) * truGeoTransform[1] + hgt_i * truGeoTransform[2];
                Geo_pt.y = truGeoTransform[3] + (wid_i + 0.5) * truGeoTransform[4] + hgt_i * truGeoTransform[5];

                //计算行列号
                double dTemp = RectGeoTransform[2] * RectGeoTransform[4] - RectGeoTransform[1] * RectGeoTransform[5];
                double dRow = (RectGeoTransform[4] * (Geo_pt.x - RectGeoTransform[0]) - RectGeoTransform[1] * (Geo_pt.y - RectGeoTransform[3])) / dTemp;
                double dCol = (RectGeoTransform[2] * (Geo_pt.y - RectGeoTransform[3]) - RectGeoTransform[5] * (Geo_pt.x - RectGeoTransform[0])) / dTemp;
                int dx = int(dCol);
                int dy = int(dRow);
                if (dy < 0 || dx < 0 || dx >= iRectWidth || dy >= iRectHeight)
                    Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                else
                {
                    float tmp_Err = float(pDataBuff[hgt_i * iDstWidth + wid_i]) - float(pDataBuffRect[dy * iRectWidth + dx]);
                    if (!isnan(tmp_Err) && abs(tmp_Err) < 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = tmp_Err;
                        mean += tmp_Err;
                        num++;
                    }
                    else if (abs(tmp_Err) > 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                    else if (isnan(tmp_Err))
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);
        //去除异常值，将3sigma外的异常值去掉，取绝对值得到误差绝对值的矩阵
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] > mean + ABNORMAL_VALUE_THRESHOLD * sigma || Error_Matrix[hgt_i * iDstWidth + wid_i] < mean - ABNORMAL_VALUE_THRESHOLD * sigma)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
                else
                {
                    Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                }
            }
        }
        mean = 0, sigma = 0, num = 0;

        //输出到同一个指定文件夹中，空值写为NaN
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    mean += Error_Matrix[hgt_i * iDstWidth + wid_i];
                    num++;
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
                }
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);

        //输出txt文件
        string Type_Str = string(RectFile).substr(string(RectFile).length() - 6, 2);
        string Print_Str = Area + "_" + Type_Str;
        printf("%s %lf %lf\n", Print_Str.c_str(), mean, sigma);

        if (is_Write_Heatmap == true)
        {
            string opt_txt1 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix.txt";
            FILE* p1;
            p1 = fopen(opt_txt1.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p1, "%lf ", Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p1, "NaN ");
                    }
                }
                fprintf(p1, "\n");
            }
            fclose(p1);

            string opt_txt2 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix_Abs.txt";
            FILE* p2;
            p2 = fopen(opt_txt2.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p2, "%lf ", Error_Matrix_Abs[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p2, "NaN ");
                    }
                }
                fprintf(p2, "\n");
            }
            fclose(p2);
        }
        RELEASE(Error_Matrix);
        RELEASE(Error_Matrix_Abs);
        RELEASE(pDataBuff);
        RELEASE(pDataBuffRect);
    }
    else if (eDT == GDT_UInt16 && eDT1 == GDT_Float32)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        unsigned short* pDataBuff = new unsigned short[iDstWidth * iDstHeight * iBandCount];
        pTruthDS->RasterIO(GF_Read, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        float* pDataBuffRect = new float[iRectWidth * iRectHeight * iBandCountRect];
        pRectDS->RasterIO(GF_Read, 0, 0, iRectWidth, iRectHeight, pDataBuffRect, iRectWidth, iRectHeight, eDT1, iBandCountRect, pBandMapRect, 0, 0, 0);

        //相减,注意以真值的图像范围为矩阵的大小，先计算平均值，注意粗差（abs > 20）及NaN值记为99999，且n--

        float* Error_Matrix = new float[iDstWidth * iDstHeight];
        float* Error_Matrix_Abs = new float[iDstWidth * iDstHeight];

        double mean = 0, sigma = 0;
        int num = 0;
        //计算图像地理坐标,若图像中某一点的行数和列数分别为：row, column,则该点的地理坐标为：
        for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
        {
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                Point3d Geo_pt;
                Geo_pt.x = truGeoTransform[0] + (wid_i + 0.5) * truGeoTransform[1] + hgt_i * truGeoTransform[2];
                Geo_pt.y = truGeoTransform[3] + (wid_i + 0.5) * truGeoTransform[4] + hgt_i * truGeoTransform[5];

                //计算行列号
                double dTemp = RectGeoTransform[2] * RectGeoTransform[4] - RectGeoTransform[1] * RectGeoTransform[5];
                double dRow = (RectGeoTransform[4] * (Geo_pt.x - RectGeoTransform[0]) - RectGeoTransform[1] * (Geo_pt.y - RectGeoTransform[3])) / dTemp;
                double dCol = (RectGeoTransform[2] * (Geo_pt.y - RectGeoTransform[3]) - RectGeoTransform[5] * (Geo_pt.x - RectGeoTransform[0])) / dTemp;
                int dx = int(dCol);
                int dy = int(dRow);
                if (dy < 0 || dx < 0 || dx >= iRectWidth || dy >= iRectHeight)
                    Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                else
                {
                    float tmp_Err = float(pDataBuff[hgt_i * iDstWidth + wid_i]) - float(pDataBuffRect[dy * iRectWidth + dx]);
                    if (!isnan(tmp_Err) && abs(tmp_Err) < 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = tmp_Err;
                        mean += tmp_Err;
                        num++;
                    }
                    else if (abs(tmp_Err) > 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                    else if (isnan(tmp_Err))
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);
        //去除异常值，将3sigma外的异常值去掉，取绝对值得到误差绝对值的矩阵
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] > mean + ABNORMAL_VALUE_THRESHOLD * sigma || Error_Matrix[hgt_i * iDstWidth + wid_i] < mean - ABNORMAL_VALUE_THRESHOLD * sigma)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
                else
                {
                    Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                }
            }
        }
        mean = 0, sigma = 0, num = 0;

        //输出到同一个指定文件夹中，空值写为NaN
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    mean += Error_Matrix[hgt_i * iDstWidth + wid_i];
                    num++;
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
                }
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);

        //输出txt文件
        string Type_Str = string(RectFile).substr(string(RectFile).length() - 6, 2);
        string Print_Str = Area + "_" + Type_Str;
        printf("%s %lf %lf\n", Print_Str.c_str(), mean, sigma);

        if (is_Write_Heatmap == true)
        {
            string opt_txt1 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix.txt";
            FILE* p1;
            p1 = fopen(opt_txt1.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p1, "%lf ", Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p1, "NaN ");
                    }
                }
                fprintf(p1, "\n");
            }
            fclose(p1);

            string opt_txt2 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix_Abs.txt";
            FILE* p2;
            p2 = fopen(opt_txt2.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p2, "%lf ", Error_Matrix_Abs[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p2, "NaN ");
                    }
                }
                fprintf(p2, "\n");
            }
            fclose(p2);
        }

        RELEASE(Error_Matrix);
        RELEASE(Error_Matrix_Abs);
        RELEASE(pDataBuff);
        RELEASE(pDataBuffRect);
    }

    else if (eDT == GDT_Float32 && eDT1 == GDT_Byte)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        float* pDataBuff = new float[iDstWidth * iDstHeight * iBandCount];
        pTruthDS->RasterIO(GF_Read, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        unsigned int* pDataBuffRect = new unsigned int[iRectWidth * iRectHeight * iBandCountRect];
        pRectDS->RasterIO(GF_Read, 0, 0, iRectWidth, iRectHeight, pDataBuffRect, iRectWidth, iRectHeight, eDT1, iBandCountRect, pBandMapRect, 0, 0, 0);

        //相减,注意以真值的图像范围为矩阵的大小，先计算平均值，注意粗差（abs > 20）及NaN值记为99999，且n--

        float* Error_Matrix = new float[iDstWidth * iDstHeight];
        float* Error_Matrix_Abs = new float[iDstWidth * iDstHeight];

        double mean = 0, sigma = 0;
        int num = 0;
        //计算图像地理坐标,若图像中某一点的行数和列数分别为：row, column,则该点的地理坐标为：
        for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
        {
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                Point3d Geo_pt;
                Geo_pt.x = truGeoTransform[0] + (wid_i + 0.5) * truGeoTransform[1] + hgt_i * truGeoTransform[2];
                Geo_pt.y = truGeoTransform[3] + (wid_i + 0.5) * truGeoTransform[4] + hgt_i * truGeoTransform[5];

                //计算行列号
                double dTemp = RectGeoTransform[2] * RectGeoTransform[4] - RectGeoTransform[1] * RectGeoTransform[5];
                double dRow = (RectGeoTransform[4] * (Geo_pt.x - RectGeoTransform[0]) - RectGeoTransform[1] * (Geo_pt.y - RectGeoTransform[3])) / dTemp;
                double dCol = (RectGeoTransform[2] * (Geo_pt.y - RectGeoTransform[3]) - RectGeoTransform[5] * (Geo_pt.x - RectGeoTransform[0])) / dTemp;
                int dx = int(dCol);
                int dy = int(dRow);
                if (dy < 0 || dx < 0 || dx >= iRectWidth || dy >= iRectHeight)
                    Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                else
                {
                    float tmp_Err = float(pDataBuff[hgt_i * iDstWidth + wid_i]) - float(pDataBuffRect[dy * iRectWidth + dx]);
                    if (!isnan(tmp_Err) && abs(tmp_Err) < 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = tmp_Err;
                        mean += tmp_Err;
                        num++;
                    }
                    else if (abs(tmp_Err) > 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                    else if (isnan(tmp_Err))
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);
        //去除异常值，将3sigma外的异常值去掉，取绝对值得到误差绝对值的矩阵
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] > mean + ABNORMAL_VALUE_THRESHOLD * sigma || Error_Matrix[hgt_i * iDstWidth + wid_i] < mean - ABNORMAL_VALUE_THRESHOLD * sigma)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
                else
                {
                    Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                }
            }
        }
        mean = 0, sigma = 0, num = 0;

        //输出到同一个指定文件夹中，空值写为NaN
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    mean += Error_Matrix[hgt_i * iDstWidth + wid_i];
                    num++;
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
                }
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);

        //输出txt文件
        string Type_Str = string(RectFile).substr(string(RectFile).length() - 6, 2);
        string Print_Str = Area + "_" + Type_Str;
        printf("%s %lf %lf\n", Print_Str.c_str(), mean, sigma);

        if (is_Write_Heatmap == true)
        {
            string opt_txt1 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix.txt";
            FILE* p1;
            p1 = fopen(opt_txt1.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p1, "%lf ", Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p1, "NaN ");
                    }
                }
                fprintf(p1, "\n");
            }
            fclose(p1);

            string opt_txt2 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix_Abs.txt";
            FILE* p2;
            p2 = fopen(opt_txt2.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p2, "%lf ", Error_Matrix_Abs[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p2, "NaN ");
                    }
                }
                fprintf(p2, "\n");
            }
            fclose(p2);
        }

        RELEASE(Error_Matrix);
        RELEASE(Error_Matrix_Abs);
        RELEASE(pDataBuff);
        RELEASE(pDataBuffRect);
    }
    else if (eDT == GDT_Byte && eDT1 == GDT_Byte)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        unsigned int* pDataBuff = new unsigned int[iDstWidth * iDstHeight * iBandCount];
        pTruthDS->RasterIO(GF_Read, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        unsigned int* pDataBuffRect = new unsigned int[iRectWidth * iRectHeight * iBandCountRect];
        pRectDS->RasterIO(GF_Read, 0, 0, iRectWidth, iRectHeight, pDataBuffRect, iRectWidth, iRectHeight, eDT1, iBandCountRect, pBandMapRect, 0, 0, 0);


        //相减,注意以真值的图像范围为矩阵的大小，先计算平均值，注意粗差（abs > 20）及NaN值记为99999，且n--

        float* Error_Matrix = new float[iDstWidth * iDstHeight];
        float* Error_Matrix_Abs = new float[iDstWidth * iDstHeight];

        double mean = 0, sigma = 0;
        int num = 0;
        //计算图像地理坐标,若图像中某一点的行数和列数分别为：row, column,则该点的地理坐标为：
        for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
        {
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                Point3d Geo_pt;
                Geo_pt.x = truGeoTransform[0] + (wid_i + 0.5) * truGeoTransform[1] + hgt_i * truGeoTransform[2];
                Geo_pt.y = truGeoTransform[3] + (wid_i + 0.5) * truGeoTransform[4] + hgt_i * truGeoTransform[5];

                //计算行列号
                double dTemp = RectGeoTransform[2] * RectGeoTransform[4] - RectGeoTransform[1] * RectGeoTransform[5];
                double dRow = (RectGeoTransform[4] * (Geo_pt.x - RectGeoTransform[0]) - RectGeoTransform[1] * (Geo_pt.y - RectGeoTransform[3])) / dTemp;
                double dCol = (RectGeoTransform[2] * (Geo_pt.y - RectGeoTransform[3]) - RectGeoTransform[5] * (Geo_pt.x - RectGeoTransform[0])) / dTemp;
                int dx = int(dCol);
                int dy = int(dRow);
                if (dy < 0 || dx < 0 || dx >= iRectWidth || dy >= iRectHeight)
                    Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                else
                {
                    float tmp_Err = float(pDataBuff[hgt_i * iDstWidth + wid_i]) - float(pDataBuffRect[dy * iRectWidth + dx]);
                    if (!isnan(tmp_Err) && abs(tmp_Err) < 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = tmp_Err;
                        mean += tmp_Err;
                        num++;
                    }
                    else if (abs(tmp_Err) > 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                    else if (isnan(tmp_Err))
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);
        //去除异常值，将3sigma外的异常值去掉，取绝对值得到误差绝对值的矩阵
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] > mean + ABNORMAL_VALUE_THRESHOLD * sigma || Error_Matrix[hgt_i * iDstWidth + wid_i] < mean - ABNORMAL_VALUE_THRESHOLD * sigma)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
                else
                {
                    Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                }
            }
        }
        mean = 0, sigma = 0, num = 0;

        //输出到同一个指定文件夹中，空值写为NaN
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    mean += Error_Matrix[hgt_i * iDstWidth + wid_i];
                    num++;
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
                }
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);

        //输出txt文件
        string Type_Str = string(RectFile).substr(string(RectFile).length() - 6, 2);
        string Print_Str = Area + "_" + Type_Str;
        printf("%s %lf %lf\n", Print_Str.c_str(), mean, sigma);

        if (is_Write_Heatmap == true)
        {
            string opt_txt1 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix.txt";
            FILE* p1;
            p1 = fopen(opt_txt1.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p1, "%lf ", Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p1, "NaN ");
                    }
                }
                fprintf(p1, "\n");
            }
            fclose(p1);

            string opt_txt2 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix_Abs.txt";
            FILE* p2;
            p2 = fopen(opt_txt2.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p2, "%lf ", Error_Matrix_Abs[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p2, "NaN ");
                    }
                }
                fprintf(p2, "\n");
            }
            fclose(p2);
        }

        RELEASE(Error_Matrix);
        RELEASE(Error_Matrix_Abs);
        RELEASE(pDataBuff);
        RELEASE(pDataBuffRect);
    }
    else if (eDT == GDT_UInt16 && eDT1 == GDT_Byte)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        unsigned short* pDataBuff = new unsigned short[iDstWidth * iDstHeight * iBandCount];
        pTruthDS->RasterIO(GF_Read, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        unsigned int* pDataBuffRect = new unsigned int[iRectWidth * iRectHeight * iBandCountRect];
        pRectDS->RasterIO(GF_Read, 0, 0, iRectWidth, iRectHeight, pDataBuffRect, iRectWidth, iRectHeight, eDT1, iBandCountRect, pBandMapRect, 0, 0, 0);

        //相减,注意以真值的图像范围为矩阵的大小，先计算平均值，注意粗差（abs > 20）及NaN值记为99999，且n--

        float* Error_Matrix = new float[iDstWidth * iDstHeight];
        float* Error_Matrix_Abs = new float[iDstWidth * iDstHeight];

        double mean = 0, sigma = 0;
        int num = 0;
        //计算图像地理坐标,若图像中某一点的行数和列数分别为：row, column,则该点的地理坐标为：
        for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
        {
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                Point3d Geo_pt;
                Geo_pt.x = truGeoTransform[0] + (wid_i + 0.5) * truGeoTransform[1] + hgt_i * truGeoTransform[2];
                Geo_pt.y = truGeoTransform[3] + (wid_i + 0.5) * truGeoTransform[4] + hgt_i * truGeoTransform[5];

                //计算行列号
                double dTemp = RectGeoTransform[2] * RectGeoTransform[4] - RectGeoTransform[1] * RectGeoTransform[5];
                double dRow = (RectGeoTransform[4] * (Geo_pt.x - RectGeoTransform[0]) - RectGeoTransform[1] * (Geo_pt.y - RectGeoTransform[3])) / dTemp;
                double dCol = (RectGeoTransform[2] * (Geo_pt.y - RectGeoTransform[3]) - RectGeoTransform[5] * (Geo_pt.x - RectGeoTransform[0])) / dTemp;
                int dx = int(dCol);
                int dy = int(dRow);
                if (dy < 0 || dx < 0 || dx >= iRectWidth || dy >= iRectHeight)
                    Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                else
                {
                    float tmp_Err = float(pDataBuff[hgt_i * iDstWidth + wid_i]) - float(pDataBuffRect[dy * iRectWidth + dx]);
                    if (!isnan(tmp_Err) && abs(tmp_Err) < 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = tmp_Err;
                        mean += tmp_Err;
                        num++;
                    }
                    else if (abs(tmp_Err) > 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                    else if (isnan(tmp_Err))
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);
        //去除异常值，将3sigma外的异常值去掉，取绝对值得到误差绝对值的矩阵
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] > mean + ABNORMAL_VALUE_THRESHOLD * sigma || Error_Matrix[hgt_i * iDstWidth + wid_i] < mean - ABNORMAL_VALUE_THRESHOLD * sigma)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
                else
                {
                    Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                }
            }
        }
        mean = 0, sigma = 0, num = 0;

        //输出到同一个指定文件夹中，空值写为NaN
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    mean += Error_Matrix[hgt_i * iDstWidth + wid_i];
                    num++;
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
                }
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);

        //输出txt文件
        string Type_Str = string(RectFile).substr(string(RectFile).length() - 6, 2);
        string Print_Str = Area + "_" + Type_Str;
        printf("%s %lf %lf\n", Print_Str.c_str(), mean, sigma);

        if (is_Write_Heatmap == true)
        {
            string opt_txt1 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix.txt";
            FILE* p1;
            p1 = fopen(opt_txt1.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p1, "%lf ", Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p1, "NaN ");
                    }
                }
                fprintf(p1, "\n");
            }
            fclose(p1);

            string opt_txt2 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix_Abs.txt";
            FILE* p2;
            p2 = fopen(opt_txt2.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p2, "%lf ", Error_Matrix_Abs[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p2, "NaN ");
                    }
                }
                fprintf(p2, "\n");
            }
            fclose(p2);
        }

        RELEASE(Error_Matrix);
        RELEASE(Error_Matrix_Abs);
        RELEASE(pDataBuff);
        RELEASE(pDataBuffRect);
    }


    else if (eDT == GDT_Float32 && eDT1 == GDT_UInt16)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        float* pDataBuff = new float[iDstWidth * iDstHeight * iBandCount];
        pTruthDS->RasterIO(GF_Read, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        unsigned short* pDataBuffRect = new unsigned short[iRectWidth * iRectHeight * iBandCountRect];
        pRectDS->RasterIO(GF_Read, 0, 0, iRectWidth, iRectHeight, pDataBuffRect, iRectWidth, iRectHeight, eDT1, iBandCountRect, pBandMapRect, 0, 0, 0);


        //相减,注意以真值的图像范围为矩阵的大小，先计算平均值，注意粗差（abs > 20）及NaN值记为99999，且n--

        float* Error_Matrix = new float[iDstWidth * iDstHeight];
        float* Error_Matrix_Abs = new float[iDstWidth * iDstHeight];

        double mean = 0, sigma = 0;
        int num = 0;
        //计算图像地理坐标,若图像中某一点的行数和列数分别为：row, column,则该点的地理坐标为：
        for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
        {
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                Point3d Geo_pt;
                Geo_pt.x = truGeoTransform[0] + (wid_i + 0.5) * truGeoTransform[1] + hgt_i * truGeoTransform[2];
                Geo_pt.y = truGeoTransform[3] + (wid_i + 0.5) * truGeoTransform[4] + hgt_i * truGeoTransform[5];

                //计算行列号
                double dTemp = RectGeoTransform[2] * RectGeoTransform[4] - RectGeoTransform[1] * RectGeoTransform[5];
                double dRow = (RectGeoTransform[4] * (Geo_pt.x - RectGeoTransform[0]) - RectGeoTransform[1] * (Geo_pt.y - RectGeoTransform[3])) / dTemp;
                double dCol = (RectGeoTransform[2] * (Geo_pt.y - RectGeoTransform[3]) - RectGeoTransform[5] * (Geo_pt.x - RectGeoTransform[0])) / dTemp;
                int dx = int(dCol);
                int dy = int(dRow);
                if (dy < 0 || dx < 0 || dx >= iRectWidth || dy >= iRectHeight)
                    Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                else
                {
                    float tmp_Err = float(pDataBuff[hgt_i * iDstWidth + wid_i]) - float(pDataBuffRect[dy * iRectWidth + dx]);
                    if (!isnan(tmp_Err) && abs(tmp_Err) < 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = tmp_Err;
                        mean += tmp_Err;
                        num++;
                    }
                    else if (abs(tmp_Err) > 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                    else if (isnan(tmp_Err))
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);
        //去除异常值，将3sigma外的异常值去掉，取绝对值得到误差绝对值的矩阵
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] > mean + ABNORMAL_VALUE_THRESHOLD * sigma || Error_Matrix[hgt_i * iDstWidth + wid_i] < mean - ABNORMAL_VALUE_THRESHOLD * sigma)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
                else
                {
                    Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                }
            }
        }
        mean = 0, sigma = 0, num = 0;

        //输出到同一个指定文件夹中，空值写为NaN
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    mean += Error_Matrix[hgt_i * iDstWidth + wid_i];
                    num++;
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
                }
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);

        //输出txt文件
        string Type_Str = string(RectFile).substr(string(RectFile).length() - 6, 2);
        string Print_Str = Area + "_" + Type_Str;
        printf("%s %lf %lf\n", Print_Str.c_str(), mean, sigma);

        if (is_Write_Heatmap == true)
        {
            string opt_txt1 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix.txt";
            FILE* p1;
            p1 = fopen(opt_txt1.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p1, "%lf ", Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p1, "NaN ");
                    }
                }
                fprintf(p1, "\n");
            }
            fclose(p1);

            string opt_txt2 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix_Abs.txt";
            FILE* p2;
            p2 = fopen(opt_txt2.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p2, "%lf ", Error_Matrix_Abs[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p2, "NaN ");
                    }
                }
                fprintf(p2, "\n");
            }
            fclose(p2);
        }

        RELEASE(Error_Matrix);
        RELEASE(Error_Matrix_Abs);
        RELEASE(pDataBuff);
        RELEASE(pDataBuffRect);
    }
    else if (eDT == GDT_Byte && eDT1 == GDT_UInt16)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        unsigned int* pDataBuff = new unsigned int[iDstWidth * iDstHeight * iBandCount];
        pTruthDS->RasterIO(GF_Read, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        unsigned short* pDataBuffRect = new unsigned short[iRectWidth * iRectHeight * iBandCountRect];
        pRectDS->RasterIO(GF_Read, 0, 0, iRectWidth, iRectHeight, pDataBuffRect, iRectWidth, iRectHeight, eDT1, iBandCountRect, pBandMapRect, 0, 0, 0);

        //相减,注意以真值的图像范围为矩阵的大小，先计算平均值，注意粗差（abs > 20）及NaN值记为99999，且n--

        float* Error_Matrix = new float[iDstWidth * iDstHeight];
        float* Error_Matrix_Abs = new float[iDstWidth * iDstHeight];

        double mean = 0, sigma = 0;
        int num = 0;
        //计算图像地理坐标,若图像中某一点的行数和列数分别为：row, column,则该点的地理坐标为：
        for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
        {
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                Point3d Geo_pt;
                Geo_pt.x = truGeoTransform[0] + (wid_i + 0.5) * truGeoTransform[1] + hgt_i * truGeoTransform[2];
                Geo_pt.y = truGeoTransform[3] + (wid_i + 0.5) * truGeoTransform[4] + hgt_i * truGeoTransform[5];

                //计算行列号
                double dTemp = RectGeoTransform[2] * RectGeoTransform[4] - RectGeoTransform[1] * RectGeoTransform[5];
                double dRow = (RectGeoTransform[4] * (Geo_pt.x - RectGeoTransform[0]) - RectGeoTransform[1] * (Geo_pt.y - RectGeoTransform[3])) / dTemp;
                double dCol = (RectGeoTransform[2] * (Geo_pt.y - RectGeoTransform[3]) - RectGeoTransform[5] * (Geo_pt.x - RectGeoTransform[0])) / dTemp;
                int dx = int(dCol);
                int dy = int(dRow);
                if (dy < 0 || dx < 0 || dx >= iRectWidth || dy >= iRectHeight)
                    Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                else
                {
                    float tmp_Err = float(pDataBuff[hgt_i * iDstWidth + wid_i]) - float(pDataBuffRect[dy * iRectWidth + dx]);
                    if (!isnan(tmp_Err) && abs(tmp_Err) < 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = tmp_Err;
                        mean += tmp_Err;
                        num++;
                    }
                    else if (abs(tmp_Err) > 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                    else if (isnan(tmp_Err))
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);
        //去除异常值，将3sigma外的异常值去掉，取绝对值得到误差绝对值的矩阵
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] > mean + ABNORMAL_VALUE_THRESHOLD * sigma || Error_Matrix[hgt_i * iDstWidth + wid_i] < mean - ABNORMAL_VALUE_THRESHOLD * sigma)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
                else
                {
                    Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                }
            }
        }
        mean = 0, sigma = 0, num = 0;

        //输出到同一个指定文件夹中，空值写为NaN
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    mean += Error_Matrix[hgt_i * iDstWidth + wid_i];
                    num++;
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
                }
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);

        //输出txt文件
        string Type_Str = string(RectFile).substr(string(RectFile).length() - 6, 2);
        string Print_Str = Area + "_" + Type_Str;
        printf("%s %lf %lf\n", Print_Str.c_str(), mean, sigma);

        if (is_Write_Heatmap == true)
        {
            string opt_txt1 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix.txt";
            FILE* p1;
            p1 = fopen(opt_txt1.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p1, "%lf ", Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p1, "NaN ");
                    }
                }
                fprintf(p1, "\n");
            }
            fclose(p1);

            string opt_txt2 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix_Abs.txt";
            FILE* p2;
            p2 = fopen(opt_txt2.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p2, "%lf ", Error_Matrix_Abs[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p2, "NaN ");
                    }
                }
                fprintf(p2, "\n");
            }
            fclose(p2);
        }

        RELEASE(Error_Matrix);
        RELEASE(Error_Matrix_Abs);
        RELEASE(pDataBuff);
        RELEASE(pDataBuffRect);
    }
    else if (eDT == GDT_UInt16 && eDT1 == GDT_UInt16)
    {
        //申请所有数据时所需要的缓存，如果图像太大应该用分块处理，以下为8bit图像示例
        unsigned short* pDataBuff = new unsigned short[iDstWidth * iDstHeight * iBandCount];
        pTruthDS->RasterIO(GF_Read, 0, 0, iDstWidth, iDstHeight, pDataBuff, iDstWidth, iDstHeight, eDT, iBandCount, pBandMap, 0, 0, 0);

        unsigned short* pDataBuffRect = new unsigned short[iRectWidth * iRectHeight * iBandCountRect];
        pRectDS->RasterIO(GF_Read, 0, 0, iRectWidth, iRectHeight, pDataBuffRect, iRectWidth, iRectHeight, eDT1, iBandCountRect, pBandMapRect, 0, 0, 0);

        //相减,注意以真值的图像范围为矩阵的大小，先计算平均值，注意粗差（abs > 20）及NaN值记为99999，且n--

        float* Error_Matrix = new float[iDstWidth * iDstHeight];
        float* Error_Matrix_Abs = new float[iDstWidth * iDstHeight];

        double mean = 0, sigma = 0;
        int num = 0;
        //计算图像地理坐标,若图像中某一点的行数和列数分别为：row, column,则该点的地理坐标为：
        for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
        {
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                Point3d Geo_pt;
                Geo_pt.x = truGeoTransform[0] + (wid_i + 0.5) * truGeoTransform[1] + hgt_i * truGeoTransform[2];
                Geo_pt.y = truGeoTransform[3] + (wid_i + 0.5) * truGeoTransform[4] + hgt_i * truGeoTransform[5];

                //计算行列号
                double dTemp = RectGeoTransform[2] * RectGeoTransform[4] - RectGeoTransform[1] * RectGeoTransform[5];
                double dRow = (RectGeoTransform[4] * (Geo_pt.x - RectGeoTransform[0]) - RectGeoTransform[1] * (Geo_pt.y - RectGeoTransform[3])) / dTemp;
                double dCol = (RectGeoTransform[2] * (Geo_pt.y - RectGeoTransform[3]) - RectGeoTransform[5] * (Geo_pt.x - RectGeoTransform[0])) / dTemp;
                int dx = int(dCol);
                int dy = int(dRow);
                if (dy < 0 || dx < 0 || dx >= iRectWidth || dy >= iRectHeight)
                    Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                else
                {
                    float tmp_Err = float(pDataBuff[hgt_i * iDstWidth + wid_i]) - float(pDataBuffRect[dy * iRectWidth + dx]);
                    if (!isnan(tmp_Err) && abs(tmp_Err) < 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = tmp_Err;
                        mean += tmp_Err;
                        num++;
                    }
                    else if (abs(tmp_Err) > 60)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                    else if (isnan(tmp_Err))
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);
        //去除异常值，将3sigma外的异常值去掉，取绝对值得到误差绝对值的矩阵
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] > mean + ABNORMAL_VALUE_THRESHOLD * sigma || Error_Matrix[hgt_i * iDstWidth + wid_i] < mean - ABNORMAL_VALUE_THRESHOLD * sigma)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        Error_Matrix[hgt_i * iDstWidth + wid_i] = 99999;
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = 99999;
                    }
                }
                else
                {
                    Error_Matrix_Abs[hgt_i * iDstWidth + wid_i] = abs(Error_Matrix[hgt_i * iDstWidth + wid_i]);
                }
            }
        }
        mean = 0, sigma = 0, num = 0;

        //输出到同一个指定文件夹中，空值写为NaN
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    mean += Error_Matrix[hgt_i * iDstWidth + wid_i];
                    num++;
                }
            }
        }
        mean /= num;
        for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
        {
            for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
            {
                if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                {
                    sigma += (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean) * (Error_Matrix[hgt_i * iDstWidth + wid_i] - mean);
                }
            }
        }
        sigma /= num;
        sigma = sqrt(sigma);

        //输出txt文件
        string Type_Str = string(RectFile).substr(string(RectFile).length() - 6, 2);
        string Print_Str = Area + "_" + Type_Str;
        printf("%s %lf %lf\n", Print_Str.c_str(), mean, sigma);

        if (is_Write_Heatmap == true)
        {
            string opt_txt1 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix.txt";
            FILE* p1;
            p1 = fopen(opt_txt1.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p1, "%lf ", Error_Matrix[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p1, "NaN ");
                    }
                }
                fprintf(p1, "\n");
            }
            fclose(p1);

            string opt_txt2 = DstDir + "\\" + Area + "_" + Type_Str + "_Error_Matrix_Abs.txt";
            FILE* p2;
            p2 = fopen(opt_txt2.c_str(), "w");
            for (int hgt_i = 0; hgt_i < iDstHeight; hgt_i++)
            {
                for (int wid_i = 0; wid_i < iDstWidth; wid_i++)
                {
                    if (Error_Matrix[hgt_i * iDstWidth + wid_i] != 99999)
                    {
                        fprintf(p2, "%lf ", Error_Matrix_Abs[hgt_i * iDstWidth + wid_i]);
                    }
                    else
                    {
                        fprintf(p2, "NaN ");
                    }
                }
                fprintf(p2, "\n");
            }
            fclose(p2);
        }

        RELEASE(Error_Matrix);
        RELEASE(Error_Matrix_Abs);
        RELEASE(pDataBuff);
        RELEASE(pDataBuffRect);
    }
    RELEASE(pBandMap);
    RELEASE(pBandMapRect);
    GDALClose((GDALDatasetH)pTruthDS);
    GDALClose((GDALDatasetH)pRectDS);
    return;
}


void Building_Mask_Elimination_Cross_Error(vector <vector<Point3d>>& BM_pt, vector <vector<Point3d>>& DSM_pt, string Mask_Path, string Mask_Type, string BM_Type, int Cut_Num)
{
    FILE* p_dom_bef;
    string DOM_txt_bef = Mask_Path + "\\" + "orthophoto_bef.txt";
    p_dom_bef = fopen(DOM_txt_bef.c_str(), "a");

    int mask_bef_size = 0;
    for (int ii = 0; ii < BM_pt.size(); ii++)
    {
        mask_bef_size += BM_pt[ii].size();
        for (int jj = 0; jj < BM_pt[ii].size(); jj++)
        {
            fprintf(p_dom_bef, "%lf %lf\n", DSM_pt[ii][jj].x, DSM_pt[ii][jj].y);
        }
    }
    fclose(p_dom_bef);

    printf(" mask_bef %d", mask_bef_size);

    if (Mask_Type == "DOM")
    {
        vector <string> files;
        GetAllFiles_tif(Mask_Path, files);

        if (files.size() != 1)
        {
            printf("若为DOM建筑物掩膜，只应有一个掩膜文件，请检查输入文件夹和掩膜类型！\n");
            return;
        }
        string filepath = Mask_Path + "\\" + files[0];

        GDALDataset* poDataset;

        GDALAllRegister(); // 注册所有的驱动程序

        // 打开文件
        poDataset = (GDALDataset*)GDALOpen(filepath.c_str(), GA_ReadOnly);
        if (poDataset == nullptr) {
            printf("无法打开掩膜文件！\n");
            return;
        }

        //获取坐标变换系数
        double trans[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
        if (poDataset->GetGeoTransform(trans) != CE_None) //影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。
        {
            printf("无法获取掩膜文件的地理信息！\n");
            GDALClose(poDataset);
            return;
        }
       trans[0] += 1.0 / 2 * trans[1];
        trans[3] += 1.0 / 2 * trans[5];

        GDALRasterBand* poBand = poDataset->GetRasterBand(1);

        for (int ii = 0; ii < BM_pt.size(); ii++)
        {
            for (int jj = 0; jj < BM_pt[ii].size(); jj++)
            {
                //计算行列号
                double dTemp = trans[2] * trans[4] - trans[1] * trans[5];
                double dRow = (trans[4] * (DSM_pt[ii][jj].x - trans[0]) - trans[1] * (DSM_pt[ii][jj].y - trans[3])) / dTemp;
                double dCol = (trans[2] * (DSM_pt[ii][jj].y - trans[3]) - trans[5] * (DSM_pt[ii][jj].x - trans[0])) / dTemp;
                int dx = int(dCol + 0.5);
                int dy = int(dRow + 0.5);
                uint8_t value;
                poBand->RasterIO(GF_Read, dx, dy, 1, 1, &value, 1, 1, GDT_Byte, 0, 0);

                //删除外点
                if (value == 1)
                {
                    BM_pt[ii].erase(BM_pt[ii].begin() + jj);
                    DSM_pt[ii].erase(DSM_pt[ii].begin() + jj);
                    jj--;
                }
            }
            if (BM_pt[ii].size() == 0)
            {
                BM_pt.erase(BM_pt.begin() + ii);
                DSM_pt.erase(DSM_pt.begin() + ii);
                ii--;
            }
        }

        // 输出一个Mask掩膜之后的所有DSM_pt点坐标的txt文件，此处为地理位置的x，y
        FILE* p_aft;
        string Mask_txt_aft =  Mask_Path + "\\Mask_DOM_" + BM_Type + ".txt";
        p_aft = fopen(Mask_txt_aft.c_str(), "w");

        for (int ii = 0; ii < BM_pt.size(); ii++)
        {
            for (int jj = 0; jj < BM_pt[ii].size(); jj++)
            {
                fprintf(p_aft, "%lf %lf\n", DSM_pt[ii][jj].x, DSM_pt[ii][jj].y);
            }
        }
        fclose(p_aft);

        // 关闭数据集
        GDALClose(poDataset);
    }

    if (Mask_Type == "BASEMAP")
    {
        vector <string> files;
        string Mask_Path_BM = Mask_Path + "\\" + BM_Type;
        GetAllFiles_tif(Mask_Path_BM, files);

        if (files.size() == 0)
        {
            printf("无掩膜文件，请检查输入文件夹！\n");
            return;
        }

        int BM_index = 0;
        int BM_max_num = files.size();
        for (int ii = 0; ii < BM_pt.size(); ii++)
        {        

            //if (BM_index * Cut_Num + Cut_Num <= ii)
            //{
            //    BM_index++;

            //}


            if (BM_pt[ii].size() == 0)
            {
                continue;
            }

            string filepath = Mask_Path_BM + "\\" + files[BM_index];

            GDALDataset* poDataset;

            GDALAllRegister(); // 注册所有的驱动程序

            // 打开文件
            poDataset = (GDALDataset*)GDALOpen(filepath.c_str(), GA_ReadOnly);
            if (poDataset == nullptr) {
                printf("无法打开掩膜文件！");
                printf("%s\n", files[BM_index].c_str());
                return;
            }

            //获取坐标变换系数
            double trans[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
            if (poDataset->GetGeoTransform(trans) != CE_None) //影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。
            {
                printf("无法获取掩膜文件的地理信息！\n");
                GDALClose(poDataset);
                return;
            }

            GDALRasterBand* poBand = poDataset->GetRasterBand(1);
            //我现在cutnum为4，有9张底图，那么我的BM_pt的数量就是4*9=36，我需要判断：在什么区间内我使用哪一张底图,BM的index是0到8，0、1、2、3..7的时候是bm0，8-15的时候是bm1
            FILE* p_bef;
            string Mask_txt_bef = Mask_Path_BM + "\\" + files[BM_index] + "bef.txt";
            p_bef = fopen(Mask_txt_bef.c_str(), "a");

            FILE* p_aft;
            string Mask_txt_aft = Mask_Path_BM + "\\" + files[BM_index] + "aft.txt";
            p_aft = fopen(Mask_txt_aft.c_str(), "a");

            for (int jj = 0; jj < BM_pt[ii].size(); jj++)
            {

                fprintf(p_bef, "%lf %lf\n", BM_pt[ii][jj].x, BM_pt[ii][jj].y);

                //计算行列号
                double dTemp = trans[2] * trans[4] - trans[1] * trans[5];
                double dRow = (trans[4] * (BM_pt[ii][jj].x - trans[0]) - trans[1] * (BM_pt[ii][jj].y - trans[3])) / dTemp;
                double dCol = (trans[2] * (BM_pt[ii][jj].y - trans[3]) - trans[5] * (BM_pt[ii][jj].x - trans[0])) / dTemp;
                int dx = int(dCol + 0.5);
                int dy = int(dRow + 0.5);
                uint8_t value;
                poBand->RasterIO(GF_Read, dx, dy, 1, 1, &value, 1, 1, GDT_Byte, 0, 0);

                //删除外点
                if (value == 1)
                {
                    BM_pt[ii].erase(BM_pt[ii].begin() + jj);
                    DSM_pt[ii].erase(DSM_pt[ii].begin() + jj);
                    jj--;
                }

                if (value == 0)
                {
                    fprintf(p_aft, "%lf %lf\n", BM_pt[ii][jj].x, BM_pt[ii][jj].y);
                }

            }
            fclose(p_bef);
            fclose(p_aft);

            // 关闭数据集
            GDALClose(poDataset);
            if (BM_index < BM_max_num - 1)
            {
                BM_index++;
            }
            else
            {
                BM_index = 0;
            }
        }

        // 输出一个Mask掩膜之后的所有DSM_pt点坐标的txt文件，此处为地理位置的x，y
        FILE* p_aft;
        string Mask_txt_aft = Mask_Path_BM + "\\Mask_BM" + "_" + BM_Type + ".txt";
        p_aft = fopen(Mask_txt_aft.c_str(), "w");

        for (int ii = 0; ii < BM_pt.size(); ii++)
        {
            if (BM_pt[ii].size() == 0)
            {
                BM_pt.erase(BM_pt.begin() + ii);
                DSM_pt.erase(DSM_pt.begin() + ii);
                ii--;
                continue;
            }
            for (int jj = 0; jj < BM_pt[ii].size(); jj++)
            {
                fprintf(p_aft, "%lf %lf\n", DSM_pt[ii][jj].x, DSM_pt[ii][jj].y);
            }
        }
        fclose(p_aft);
    }
    int mask_aft_size = 0;

    FILE* p_dom_aft;
    string DOM_txt_aft = Mask_Path + "\\" + "orthophoto_aft.txt";
    p_dom_aft = fopen(DOM_txt_aft.c_str(), "a");

    for (int ii = 0; ii < BM_pt.size(); ii++)
    {
        mask_aft_size += BM_pt[ii].size();
        for (int jj = 0; jj < BM_pt[ii].size(); jj++)
        {
            fprintf(p_dom_aft, "%lf %lf\n", DSM_pt[ii][jj].x, DSM_pt[ii][jj].y);
        }
    }
    fclose(p_dom_aft);

    printf(" mask_aft %d ", mask_aft_size);

}


// 假设这个函数可以获取掩膜图像中特定像素的值
int GetMaskValue(GDALDataset* poMaskDS, double geoX, double geoY) {
    // 获取地理转换参数
    double adfGeoTransform[6];
    if (poMaskDS->GetGeoTransform(adfGeoTransform) != CE_None) {
        // 错误处理
        return -1;
    }

    // 将地理坐标转换为像素坐标
    int xPixel = static_cast<int>((geoX - adfGeoTransform[0]) / adfGeoTransform[1]);
    int yPixel = static_cast<int>((geoY - adfGeoTransform[3]) / adfGeoTransform[5]);

    // 获取第一个波段（通常掩膜文件只有一个波段）
    GDALRasterBand* poBand = poMaskDS->GetRasterBand(1);
    if (poBand == nullptr) {
        // 错误处理
        return -1;
    }

    // 读取指定像素的值
    uint8_t nMaskValue;
    if (poBand->RasterIO(GF_Read, xPixel, yPixel, 1, 1, &nMaskValue, 1, 1, GDT_Byte, 0, 0) != CE_None) {
        // 错误处理
        return -1;
    }

    return int(nMaskValue);
}

double CalculateConfidenceForCell(GDALDataset* poMaskDS, double cellGeoX, double cellGeoY, double cellSizeX, double cellSizeY) {
    int maskOneCount = 0;
    int maskValidCount = 0;

    // 获取掩膜文件的地理转换参数
    double maskGeoTransform[6];
    if (poMaskDS->GetGeoTransform(maskGeoTransform) != CE_None) {
        // 错误处理
        return -1;
    }

    // 获取掩膜文件的空间范围
    double maskMinX = maskGeoTransform[0];
    double maskMaxX = maskGeoTransform[0] + poMaskDS->GetRasterXSize() * maskGeoTransform[1];
    double maskMaxY = maskGeoTransform[3];
    double maskMinY = maskGeoTransform[3] + poMaskDS->GetRasterYSize() * maskGeoTransform[5]; // 通常为负值

    // 确定遍历的起始和结束坐标
    double startX = std::max(cellGeoX, maskMinX);
    double endX = std::min(cellGeoX + cellSizeX, maskMaxX);
    double startY = std::max(cellGeoY - cellSizeY, maskMinY);
    double endY = std::min(cellGeoY, maskMaxY);

    // 计算掩膜文件的像素分辨率
    double maskPixelSizeX = maskGeoTransform[1];
    double maskPixelSizeY = -maskGeoTransform[5]; // 通常为负值

    // 遍历交集区域
    for (double y = startY; y < endY; y += maskPixelSizeY) {
        for (double x = startX; x < endX; x += maskPixelSizeX) {
            int maskValue = GetMaskValue(poMaskDS, x, y);
            if (maskValue == 1 || maskValue == 0) {
                maskValidCount++;
                if (maskValue == 1) {
                    maskOneCount++;
                }
            }
        }
    }

    // 计算置信度
    return (maskValidCount > 0) ? (1 - static_cast<double>(maskOneCount) / maskValidCount) : 1.0;
}


void Elevation_Weight_Calculation(vector<vector<Point3d>>& BM_pt, vector<vector<Point3d>>& DSM_pt, string Mask_Path, string BM_Type, string dir_path, vector<vector<double>>& finalConfidenceMatrix) {
    vector<string> tif_dem;
    string BM_dem_path = dir_path + "\\" + "BASEMAP" + "\\" + "SRTM";
    GetAllFiles_tif(BM_dem_path, tif_dem);

    vector<string> files;
    Mask_Path = Mask_Path + "\\" + BM_Type;
    GetAllFiles_tif(Mask_Path, files);

    if (files.size() == 0) {
        printf("无掩膜文件，请检查输入文件夹！\n");
        return;
    }

    GDALAllRegister();
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名

    // 读取SRTM数据
    GDALDataset* poDataset = (GDALDataset*)GDALOpen((BM_dem_path + "\\" + tif_dem[0]).c_str(), GA_ReadOnly);
    if (poDataset == NULL) {
        cout << "无法读取SRTM数据！" << endl;
        return;
    }

    int nXSize = poDataset->GetRasterXSize();
    int nYSize = poDataset->GetRasterYSize();
    double adfGeoTransform[6];
    poDataset->GetGeoTransform(adfGeoTransform);
    double cellSizeX = adfGeoTransform[1]; // 一个格子的宽度
    double cellSizeY = -adfGeoTransform[5]; // 一个格子的高度（通常为负值）


    //vector<vector<double>> sharedConfidenceMatrix(nYSize, vector<double>(nXSize, 0.0));

//#pragma omp parallel
//    {
//        vector<vector<double>> localConfidenceMatrix(nYSize, vector<double>(nXSize, 0.0));
//
//#pragma omp for nowait
//        for (int fileIndex = 0; fileIndex < files.size(); ++fileIndex) {
//            const string& file = files[fileIndex];
//            GDALDataset* poMaskDS = (GDALDataset*)GDALOpen((Mask_Path + "\\" + file).c_str(), GA_ReadOnly);
//            if (poMaskDS == NULL) {
//                cout << "无法读取掩膜文件：" << file << endl;
//                continue;
//            }
//
//            double maskGeoTransform[6];
//            if (poMaskDS->GetGeoTransform(maskGeoTransform) != CE_None) {
//                GDALClose(poMaskDS);
//                continue;
//            }
//
//            double maskMinX = maskGeoTransform[0];
//            double maskMaxX = maskGeoTransform[0] + poMaskDS->GetRasterXSize() * maskGeoTransform[1];
//            double maskMaxY = maskGeoTransform[3];
//            double maskMinY = maskGeoTransform[3] + poMaskDS->GetRasterYSize() * maskGeoTransform[5];
//
//            for (int i = 0; i < nYSize; ++i) {
//                for (int j = 0; j < nXSize; ++j) {
//                    double cellGeoX = adfGeoTransform[0] + j * cellSizeX;
//                    double cellGeoY = adfGeoTransform[3] - i * cellSizeY;
//
//                    if (cellGeoX < maskMaxX && cellGeoX + cellSizeX > maskMinX &&
//                        cellGeoY < maskMaxY && cellGeoY - cellSizeY > maskMinY) {
//                        localConfidenceMatrix[i][j] += CalculateConfidenceForCell(poMaskDS, cellGeoX, cellGeoY, cellSizeX, cellSizeY);
//                    }
//                    else {
//                        localConfidenceMatrix[i][j] += 1.0; // 赋予默认置信度
//                    }
//                }
//            }
//            GDALClose(poMaskDS);
//        }
//
//        // 将局部矩阵累加到共享矩阵中
//#pragma omp critical
//        for (int i = 0; i < nYSize; ++i) {
//            for (int j = 0; j < nXSize; ++j) {
//                sharedConfidenceMatrix[i][j] += localConfidenceMatrix[i][j];
//            }
//        }
//    }


    // 存储每张掩膜的置信度矩阵
    // 为外层向量分配大小，每个元素是一个vector<double>
    finalConfidenceMatrix.resize(nYSize);

    // 为每个内层向量（即每一行）分配大小和初始值
    for (int i = 0; i < nYSize; ++i) {
        finalConfidenceMatrix[i].resize(nXSize, 0.0);
    }

    for (const string& file : files) {
        GDALAllRegister();
        CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名

        GDALDataset* poMaskDS = (GDALDataset*)GDALOpen((Mask_Path + "\\" + file).c_str(), GA_ReadOnly);
        if (poMaskDS == NULL) {
            cout << "无法读取掩膜文件：" << file << endl;
            continue;
        }

        // 获取掩膜文件的地理转换参数
        double maskGeoTransform[6];
        if (poMaskDS->GetGeoTransform(maskGeoTransform) != CE_None) {
            // 错误处理
            GDALClose(poMaskDS);
            continue;
        }

        // 获取掩膜文件的空间范围
        double maskMinX = maskGeoTransform[0];
        double maskMaxX = maskGeoTransform[0] + poMaskDS->GetRasterXSize() * maskGeoTransform[1];
        double maskMaxY = maskGeoTransform[3];
        double maskMinY = maskGeoTransform[3] + poMaskDS->GetRasterYSize() * maskGeoTransform[5]; // 通常为负值

        // 计算当前掩膜的置信度
        for (int i = 0; i < nYSize; ++i) {
            // assert(i < finalConfidenceMatrix.size());
            for (int j = 0; j < nXSize; ++j) {
                // assert(j < finalConfidenceMatrix[i].size());
                double cellGeoX = adfGeoTransform[0] + j * cellSizeX;
                double cellGeoY = adfGeoTransform[3] - i * cellSizeY;

                // 检查SRTM方格是否在掩膜范围内
                if (cellGeoX < maskMaxX && cellGeoX + cellSizeX > maskMinX &&
                    cellGeoY < maskMaxY && cellGeoY - cellSizeY > maskMinY) {
                    finalConfidenceMatrix[i][j] += CalculateConfidenceForCell(poMaskDS, cellGeoX, cellGeoY, cellSizeX, cellSizeY);
                }
                else {
                    // 方格不在掩膜范围内，可以选择跳过或赋予默认置信度
                    finalConfidenceMatrix[i][j] += 1.0; // 例如，赋予默认置信度1.0
                }
            }
        }

        GDALClose(poMaskDS);

    }


        // 计算每个格子的平均置信度
        // 确保 finalConfidenceMatrix 的尺寸正确
        // assert(finalConfidenceMatrix.size() == nYSize);
        //for (const auto& row : finalConfidenceMatrix) {
        //    assert(row.size() == nXSize);
        //}
    for (auto& row : finalConfidenceMatrix) {
        for (double& value : row) {
            value /= files.size();
        }
    }

    //// 将最终置信度应用到BM_pt
    //// // 初始化 Elevation_Weight 矩阵，使其与 BM_pt 一样大小，并填充为0
    //Elevation_Weight.resize(BM_pt.size());
    //for (size_t i = 0; i < BM_pt.size(); ++i) {
    //    Elevation_Weight[i].resize(BM_pt[i].size(), 0.0);
    //}
    //// 在应用置信度到 BM_pt 之前加上断言
    ////assert(Elevation_Weight.size() == BM_pt.size());
    //for (size_t i = 0; i < BM_pt.size(); ++i) {
    //    for (size_t j = 0; j < BM_pt[i].size(); ++j) {
    //        int xIndex = static_cast<int>((BM_pt[i][j].x - adfGeoTransform[0]) / cellSizeX);
    //        int yIndex = static_cast<int>((BM_pt[i][j].y - adfGeoTransform[3]) / -cellSizeY);
    //        if (finalConfidenceMatrix[yIndex][xIndex] > 0.8)
    //        {
    //            Elevation_Weight[i][j] = finalConfidenceMatrix[yIndex][xIndex];
    //        }
    //    }
    //}
    GDALDestroyDriverManager();
}


void GetAllFiles_txt2GeoInfo_BM_and_DSM(string dir_path, string BM_type, vector <vector <Point3d>>& BM_All_Points_input, vector <vector <Point3d>>& DSM_All_Points_input)
{
    _int64 hFile = 0;
    //文件信息结构体
    vector<string> txt_files_BM;
    vector<string> txt_files_DSM;
    string path = dir_path + "\\" + "MATCH_RESULT" + "\\" + BM_type;
    struct _finddata_t fileinfo;
    string p;
    if ((hFile = _findfirst(p.assign(path).append("\\*kpt1.txt").c_str(), &fileinfo)) != -1)
    {
        do
        {
            txt_files_BM.push_back(fileinfo.name);
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }

    if ((hFile = _findfirst(p.assign(path).append("\\*kpt0.txt").c_str(), &fileinfo)) != -1)
    {
        do
        {
            txt_files_DSM.push_back(fileinfo.name);
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }

    // 底图文件读取初始化
    vector <string> tif_files;
    vector <string> tif_dem;
    string BM_tif_path = dir_path + "\\" + "BASEMAP" + "\\" + BM_type;
    string BM_dem_path = dir_path + "\\" + "BASEMAP" + "\\" + "SRTM";
    GetAllFiles_tif(BM_tif_path, tif_files);
    for (int bm_i = 0; bm_i < tif_files.size(); bm_i++)
    {
        if (tif_files[bm_i].find("orthophoto.tif") != string::npos)
        {
            tif_files.erase(tif_files.begin() + bm_i);
            break;
        }
    }

    GetAllFiles_tif(BM_dem_path, tif_dem);
    vector <vector <Point3d>> BM_All_Points(txt_files_BM.size());

    // DSM、DOM文件读取初始化
    vector <string> tif_dom;
    vector <string> tif_dsm;
    vector <string> dom_tfw;
    vector <string> dsm_tfw;
    string DOM_path = dir_path + "\\" + "DOM";
    string DSM_path = dir_path + "\\" + "DSM";
    GetAllFiles_tif(DOM_path, tif_dom);
    GetAllFiles_tif(DSM_path, tif_dsm);
    GetAllFiles_tfw(DOM_path, dom_tfw);
    GetAllFiles_tfw(DSM_path, dsm_tfw);
    vector <vector <Point3d>> DSM_All_Points(txt_files_DSM.size());

    GDALAllRegister();//注册所有的驱动
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名

    // 开始循环
    for (int i = 0; i < txt_files_BM.size(); i++)
    {
        // BASEMAP
        Point3d temp_pt_BM;
        FILE* p_BM;
        string path_txt_BM = path + "\\" + txt_files_BM[i];
        p_BM = fopen(path_txt_BM.c_str(), "r");


        string path_tif_BM = BM_tif_path + "\\" + tif_files[i];
        //加载tif数据
        GDALDataset* poDataset_BM = (GDALDataset*)GDALOpen(path_tif_BM.c_str(), GA_ReadOnly);//GA_Update和GA_ReadOnly两种模式
        if (poDataset_BM == NULL) {
            std::cout << "指定的文件不能打开!" << std::endl;
            exit(0);
        }

        //获取坐标变换系数
        double transBM[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
        poDataset_BM->GetGeoTransform(transBM);//影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。

        string path_dem = BM_dem_path + "\\" + tif_dem[0];
        GDALDataset* poDataset_DEM = (GDALDataset*)GDALOpen(path_dem.c_str(), GA_ReadOnly);
        if (poDataset_DEM == NULL) {
            std::cout << "指定的文件不能打开!" << std::endl;
            exit(0);
        }

        //获取dem坐标变换系数
        double transdem[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
        poDataset_DEM->GetGeoTransform(transdem);//影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。
        transdem[0] += 1.0 / 2 * transdem[1];
        transdem[3] += 1.0 / 2 * transdem[5];
        double x_BM, y_BM;



        //DOM+DSM
        Point3d temp_pt_DOM;
        FILE* p_DOM;
        string path_txt_DOM = path + "\\" + txt_files_DSM[i];
        p_DOM = fopen(path_txt_DOM.c_str(), "r");


        string path_tif_DOM = DOM_path + "\\" + tif_dom[0];
        //加载tif数据
        GDALDataset* poDataset_DOM = (GDALDataset*)GDALOpen(path_tif_DOM.c_str(), GA_ReadOnly);//GA_Update和GA_ReadOnly两种模式
        if (poDataset_DOM == NULL) {
            std::cout << "指定的文件不能打开!" << std::endl;
            exit(0);
        }

        //获取dom坐标变换系数
        double trans_DOM[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
        poDataset_DOM->GetGeoTransform(trans_DOM);//影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。

        string path_dsm = DSM_path + "\\" + tif_dsm[0];
        GDALDataset* poDataset_DSM = (GDALDataset*)GDALOpen(path_dsm.c_str(), GA_ReadOnly);//GA_Update和GA_ReadOnly两种模式
        if (poDataset_DSM == NULL) {
            std::cout << "指定的文件不能打开!" << std::endl;
            exit(0);
        }

        //获取dem坐标变换系数
        double transdsm[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
        poDataset_DSM->GetGeoTransform(transdsm);//影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。

        double x_DOM, y_DOM;



        // 开始读取txt文件
        while (fscanf(p_BM, "%lf %lf", &x_BM, &y_BM) != EOF && fscanf(p_DOM, "%lf %lf", &x_DOM, &y_DOM) != EOF)
        {

            //计算图像地理坐标,若图像中某一点的行数和列数分别为：row, column,则该点的地理坐标为：
            temp_pt_BM.x = transBM[0] + x_BM * transBM[1] + y_BM * transBM[2];
            temp_pt_BM.y = transBM[3] + x_BM * transBM[4] + y_BM * transBM[5];

            //计算行列号
            double dTemp_BM = transdem[2] * transdem[4] - transdem[1] * transdem[5];
            double dRow_BM = (transdem[4] * (temp_pt_BM.x - transdem[0]) - transdem[1] * (temp_pt_BM.y - transdem[3])) / dTemp_BM;
            double dCol_BM = (transdem[2] * (temp_pt_BM.y - transdem[3]) - transdem[5] * (temp_pt_BM.x - transdem[0])) / dTemp_BM;
            int dx_BM = int(dCol_BM + 0);
            int dy_BM = int(dRow_BM + 0);//这里出问题

            //计算图像地理坐标,若图像中某一点的行数和列数分别为：row, column,则该点的地理坐标为：
            temp_pt_DOM.x = trans_DOM[0] + x_DOM * trans_DOM[1] + y_DOM * trans_DOM[2];
            temp_pt_DOM.y = trans_DOM[3] + x_DOM * trans_DOM[4] + y_DOM * trans_DOM[5];

            //计算行列号
            double dTemp_DOM = transdsm[2] * transdsm[4] - transdsm[1] * transdsm[5];
            double dRow_DOM = (transdsm[4] * (temp_pt_DOM.x - transdsm[0]) - transdsm[1] * (temp_pt_DOM.y - transdsm[3])) / dTemp_DOM;
            double dCol_DOM = (transdsm[2] * (temp_pt_DOM.y - transdsm[3]) - transdsm[5] * (temp_pt_DOM.x - transdsm[0])) / dTemp_DOM;
            int dx_DOM = int(dCol_DOM + 0);
            int dy_DOM = int(dRow_DOM + 0);

            // 一般高程数据类型就是float，因此不再进行判断，如果为了严谨应该判断类型
            GDALRasterBand* poBand_DEM = poDataset_DEM->GetRasterBand(1);
            float* buffer_DEM = (float*)CPLMalloc(sizeof(float) * 1);
            if (dx_BM < 0 || dy_BM < 0 || dx_BM >= poDataset_DEM->GetRasterXSize() || dy_BM >= poDataset_DEM->GetRasterXSize())
            {
                continue;
            }

            GDALRasterBand* poBand_DSM = poDataset_DSM->GetRasterBand(1);
            float* buffer_DSM = (float*)CPLMalloc(sizeof(float) * 1);
            if (dx_DOM < 0 || dy_DOM < 0 || dx_DOM >= poDataset_DSM->GetRasterXSize() || dy_DOM >= poDataset_DSM->GetRasterXSize())
            {
                continue;
            }
            if (GDALRasterIO(poBand_DEM, GF_Read, dx_BM, dy_BM, 1, 1, buffer_DEM, 1, 1, GDT_Float32, 0, 0) != CE_None) exit(0);
            if (GDALRasterIO(poBand_DSM, GF_Read, dx_DOM, dy_DOM, 1, 1, buffer_DSM, 1, 1, GDT_Float32, 0, 0) != CE_None) exit(0);

            // 赋值像素高程
            temp_pt_BM.z = double(*buffer_DEM);
            BM_All_Points[i].push_back(temp_pt_BM);
            CPLFree(buffer_DEM);

            // 赋值像素高程
            temp_pt_DOM.z = double(*buffer_DSM);
            DSM_All_Points[i].push_back(temp_pt_DOM);
            CPLFree(buffer_DSM);
        }
        GDALClose(poDataset_BM);
        GDALClose(poDataset_DEM);
        GDALClose(poDataset_DOM);
        GDALClose(poDataset_DSM);
        fclose(p_BM);
        fclose(p_DOM);
    }
    BM_All_Points_input = move(BM_All_Points);
    DSM_All_Points_input = move(DSM_All_Points);
}


void Image_Chunking_Process_Para_Adjust(string All_Area_Dir, int Cut_Num, string exe_path, bool is_Feature_Matching, bool is_Geometric_Correction, bool is_Precision_Evaluation, bool is_Building_Mask, string Mask_Type, bool is_Elevation_Weight, bool is_RANSAC_Iter, bool is_Basemap_Num)
{
    if (is_RANSAC_Iter == true && is_Basemap_Num == false)
    {
        //第一步，将分块后的影像放入正确的文件夹中
        vector <string> Area_Folder;
        GetFolder(All_Area_Dir, Area_Folder);
        //if (Area_Folder.size() == 1)//需要从头开始，即把Guangzhou_is_Chunked去掉
        //{
        //printf("正在进行分块操作\n");
        //string path_ARC_BING_MB = All_Area_Dir + "\\" + Area_Folder[0] + "\\BASEMAP" + "\\ARC_BING_MB";
        //string path_GE = All_Area_Dir + "\\" + Area_Folder[0] + "\\BASEMAP" + "\\GE";
        //string path_SRTM = All_Area_Dir + "\\" + Area_Folder[0] + "\\BASEMAP" + "\\SRTM";
        //if (0 != _access((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked").c_str(), 0))
        //{
        //        // if this folder not exist, create a new one.
        //    _mkdir((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked").c_str());   // 返回 0 表示创建成功，-1 表示失败
        //}
        //Image_Chunking(path_ARC_BING_MB, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "ARC_BING_MB");
        //Image_Chunking(path_GE, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "GE");
        //Image_Chunking(path_SRTM, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "SRTM");// SRTM 不做裁剪
        ////分块DOM和DSM，并把DOM放到对应的底图文件夹中
        //string path_DOM_DSM = All_Area_Dir + "\\" + Area_Folder[0];
        //Image_Chunking(path_DOM_DSM, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "DOM_DSM");

        //printf("分块操作已完成\n");
        //}
        //else
        //{
        //    printf("无法完成底图分块操作，由于有一个以上文件夹处于数据所在路径下，默认已完成分块进行后续操作\n");
        //}

        //第二步，将底图BASEMAP文件夹中的两个文件夹中的文件写成分别命名的txt
        vector <string> Chunked_Folder;
        GetFolder(All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Chunked_Folder);

        if (is_Geometric_Correction == true)
        {
            printf("正在进行DSM和DOM的几何校正......\n");

            // 1、加入三个方向的误差，xyz方向，5到10m，伪随机误差即可

            // 定义一个结构体来存储XYZ三个方向上的误差
            vector<Point3d> Errors;
            vector<Point3d> Paras; // 校正后计算得到的误差

            // 初始化随机数生成器
            std::random_device rd;
            std::mt19937 gen(rd());
            // 定义误差范围为5到10米
            std::uniform_real_distribution<> dis(20.0, 30.0);

            // 生成十组误差
            for (int i = 0; i < 1; ++i) {
                Point3d error_n;
                error_n.x = dis(gen);
                error_n.y = dis(gen);
                error_n.z = dis(gen);
                Errors.push_back(error_n);
            }

            vector<vector<double>> finalConfidenceMatrix;

            int ransac_times = 100;
            double ransac_step = 0.1;
            int iter_times = 100;
            double iter_step = 0.05;
            for (int error_i = 0; error_i < Errors.size(); error_i++)
            {

                // 3、for循环修改RANSAC_THRESHOLD和WEIGHT_ITERATION_SIGMA_BEF
                for (int ransac_i = 0; ransac_i < ransac_times; ransac_i++)
                {
                    vector <vector <Point3d>> BM_All_Points_GE_All;
                    vector <vector <Point3d>> BM_All_Points_DSM_GE_All;

                    //这个for循环的目的是将所有分块区域的匹配点都加入到同一个vector数组中，从而可以达到和整体平差基本一样的结果
                    for (int folder_area = 0; folder_area < Chunked_Folder.size(); folder_area++)
                    {
                        // 2、修改读取点位置的函数，传入三个方向的误差，进行添加误差处理
                        vector <vector <Point3d>> BM_All_Points_GE;
                        vector <vector <Point3d>> BM_All_Points_DSM_GE;
                        GetAllFiles_txt2GeoInfo_BM_and_DSM_Para_Adjust(All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[folder_area], "GE", BM_All_Points_GE, BM_All_Points_DSM_GE, Errors[error_i].x, Errors[error_i].y, Errors[error_i].z);
                        for (int i = 0; i < BM_All_Points_GE.size(); i++)
                            BM_All_Points_GE_All.push_back(BM_All_Points_GE[i]);
                        for (int i = 0; i < BM_All_Points_DSM_GE.size(); i++)
                            BM_All_Points_DSM_GE_All.push_back(BM_All_Points_DSM_GE[i]);
                    }

                    Point3d Model_Translation_Parameter_GE_Ransac;
                    Point3d Model_Translation_Parameter_GE_Iter;


                    double RANSAC_THRESHOLD_i = ransac_step + ransac_step * ransac_i; //0.2到10


                    // printf("正在进行GE的粗校正......\n");
                    RANSAC_Point(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All, 6000, RANSAC_THRESHOLD_i, 0.995, RANSAC_RATIO, Model_Translation_Parameter_GE_Ransac, string("GE"), All_Area_Dir + "\\" + Area_Folder[0]);
                    double x_GE = Model_Translation_Parameter_GE_Ransac.x;
                    double y_GE = Model_Translation_Parameter_GE_Ransac.y;
                    double z_GE = Model_Translation_Parameter_GE_Ransac.z;
                    // printf("已完成GE的粗校正\n");

                    Point3d Model_Params;
                    Model_Params.x = x_GE;
                    Model_Params.y = y_GE;
                    Model_Params.z = z_GE;

                    calculateGeoPositioningAccuracy("Coarse Correction", Model_Params, All_Area_Dir + "\\Truth_GCP.txt", All_Area_Dir + "\\Unrectified_GCP.txt");

                    if (is_Building_Mask == true)
                    {
                        // printf("正在进行基于%s建筑物掩膜的粗差剔除......\n", Mask_Type.c_str());
                        string Mask_Path = All_Area_Dir + "\\" + Area_Folder[0] + "\\Building_Mask";
                        Building_Mask_Elimination_Cross_Error(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All, Mask_Path, Mask_Type, "GE", Cut_Num);
                        // printf("已完成基于%s建筑物掩膜的粗差剔除\n", Mask_Type.c_str());
                    }

                    for (int iter_i = 0; iter_i < iter_times; iter_i++)//17
                    {
                        double WEIGHT_ITERATION_SIGMA_BEF_i = iter_step + iter_step * iter_i; //0.15到6
                        // printf("正在进行GE的精校正......\n");
                        if (is_Elevation_Weight == true && ransac_i == 0 && iter_i == 0 && error_i == 0)
                        {
                            // 计算高程的权重（置信度）
                            // printf("正在进行高程赋权\n");
                            string Mask_Path = All_Area_Dir + "\\" + Area_Folder[0] + "\\Building_Mask";

                            Elevation_Weight_Calculation(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All, Mask_Path, "GE", All_Area_Dir + "\\" + Area_Folder[0], finalConfidenceMatrix);
                            // printf("已完成高程赋权\n");
                        }

                        Weight_Iteration_Point_Elevation(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All, x_GE, y_GE, z_GE, WEIGHT_ITERATION_SIGMA_BEF_i, Model_Translation_Parameter_GE_Iter, finalConfidenceMatrix, All_Area_Dir + "\\" + Area_Folder[0]);

                        double x_GE_iter = Model_Translation_Parameter_GE_Iter.x;
                        double y_GE_iter = Model_Translation_Parameter_GE_Iter.y;
                        double z_GE_iter = Model_Translation_Parameter_GE_Iter.z;

                        Model_Params.x = x_GE_iter;
                        Model_Params.y = y_GE_iter;
                        Model_Params.z = z_GE_iter;

                        calculateGeoPositioningAccuracy("Coarse+Fine Correction", Model_Params, All_Area_Dir + "\\Truth_GCP.txt", All_Area_Dir + "\\Unrectified_GCP.txt");


                        // printf("已完成GE的精校正\n");
                        printf("x %lf y %lf z %lf delta_x %lf delta_y %lf h_RMSE %lf v_RMSE %lf RANSAC_THRESHOLD_i %lf WEIGHT_ITERATION_SIGMA_BEF_i %lf\n", Errors[error_i].x, Errors[error_i].y, Errors[error_i].z, abs(Errors[error_i].x + x_GE_iter), abs(Errors[error_i].y + y_GE_iter), sqrt(pow(abs(Errors[error_i].x + x_GE_iter), 2) + pow(abs(Errors[error_i].y + y_GE_iter), 2)), abs(Errors[error_i].z + z_GE_iter), ransac_step + ransac_step * ransac_i, iter_step + iter_step * iter_i);
                        Point3d temp_para;
                        temp_para.x = x_GE_iter;
                        temp_para.y = y_GE_iter;
                        temp_para.z = z_GE_iter;
                        Paras.push_back(temp_para);
                    }
                }

            }


            // 4、计算校正参数，和添加的误差进行对比，从而得到校正误差，计算水平方向RMSE、垂直方向RMSE以及总的RMSE

            FILE* rmse1;
            rmse1 = fopen((All_Area_Dir + "\\" + "RMSE_Results.txt").c_str(), "w");
            // 计算RMSE
            for (int error_i = 0; error_i < Errors.size(); error_i++) {
                for (int ransac_i = 0; ransac_i < ransac_times; ransac_i++)//10
                {
                    for (int iter_i = 0; iter_i < iter_times; iter_i++)//17
                    {
                        fprintf(rmse1, "x %lf y %lf z %lf delta_x %lf delta_y %lf h_RMSE %lf v_RMSE %lf RANSAC_THRESHOLD_i %lf WEIGHT_ITERATION_SIGMA_BEF_i %lf\n", Errors[error_i].x, Errors[error_i].y, Errors[error_i].z, abs(Errors[error_i].x + Paras[iter_i + ransac_i * iter_times].x), abs(Errors[error_i].y + Paras[iter_i + ransac_i * iter_times].y), sqrt(pow(abs(Errors[error_i].x + Paras[iter_i + ransac_i * iter_times].x), 2) + pow(abs(Errors[error_i].y + Paras[iter_i + ransac_i * iter_times].y), 2)), abs(Errors[error_i].z + Paras[iter_i + ransac_i * iter_times].z), ransac_step + ransac_step * ransac_i, iter_step + iter_step * iter_i);
                    }
                }

            }
            fclose(rmse1); // 关闭文件
        }
    }
    else if (is_Basemap_Num == true && is_RANSAC_Iter == false)
    {
        //第一步，将分块后的影像放入正确的文件夹中
        vector <string> Area_Folder;
        GetFolder(All_Area_Dir, Area_Folder);
        //if (Area_Folder.size() == 1)//需要从头开始，即把Guangzhou_is_Chunked去掉
        //{
        //printf("正在进行分块操作\n");
        //string path_ARC_BING_MB = All_Area_Dir + "\\" + Area_Folder[0] + "\\BASEMAP" + "\\ARC_BING_MB";
        //string path_GE = All_Area_Dir + "\\" + Area_Folder[0] + "\\BASEMAP" + "\\GE";
        //string path_SRTM = All_Area_Dir + "\\" + Area_Folder[0] + "\\BASEMAP" + "\\SRTM";
        //if (0 != _access((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked").c_str(), 0))
        //{
        //        // if this folder not exist, create a new one.
        //    _mkdir((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked").c_str());   // 返回 0 表示创建成功，-1 表示失败
        //}
        //Image_Chunking(path_ARC_BING_MB, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "ARC_BING_MB");
        //Image_Chunking(path_GE, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "GE");
        //Image_Chunking(path_SRTM, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "SRTM");// SRTM 不做裁剪
        ////分块DOM和DSM，并把DOM放到对应的底图文件夹中
        //string path_DOM_DSM = All_Area_Dir + "\\" + Area_Folder[0];
        //Image_Chunking(path_DOM_DSM, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "DOM_DSM");

        //printf("分块操作已完成\n");
        //}
        //else
        //{
        //    printf("无法完成底图分块操作，由于有一个以上文件夹处于数据所在路径下，默认已完成分块进行后续操作\n");
        //}

        //第二步，将底图BASEMAP文件夹中的两个文件夹中的文件写成分别命名的txt
        vector <string> Chunked_Folder;
        GetFolder(All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Chunked_Folder);

        if (is_Geometric_Correction == true)
        {
            printf("正在进行DSM和DOM的几何校正......\n");

            // 1、加入三个方向的误差，xyz方向，5到10m，伪随机误差即可

            // 定义一个结构体来存储XYZ三个方向上的误差
            vector<Point3d> Errors;
            vector<Point3d> Paras; // 校正后计算得到的误差

            // 初始化随机数生成器
            std::random_device rd;
            std::mt19937 gen(rd());
            // 定义误差范围为5到10米
            // std::uniform_real_distribution<> dis(20.0, 30.0);
            std::uniform_real_distribution<> dis(0.0, 0.0);
            // 生成十组误差
            for (int i = 0; i < 1; ++i) {
                Point3d error_n;
                error_n.x = dis(gen);
                error_n.y = dis(gen);
                error_n.z = dis(gen);
                Errors.push_back(error_n);
            }

            int max_basemap_num = 11; //此处定义最大底图
            int basemap_random_times = 10;
            
            srand((unsigned)time(0)); // 设置随机数种子，且仅设置一次
            vector <set <vector<int>>> all_basemap_group;

            for (int max_basemap_num_i = 1; max_basemap_num_i < max_basemap_num; max_basemap_num_i++) {
                set<vector<int>> basemap_group; // 每次外循环开始时重置

                while (basemap_group.size() < basemap_random_times) {
                    vector<int> basemap_group_i;
                    set<int> uniqueNumbers;

                    while (uniqueNumbers.size() < max_basemap_num_i) {
                        int randomNumber = rand() % max_basemap_num;
                        uniqueNumbers.insert(randomNumber);
                    }

                    basemap_group_i.assign(uniqueNumbers.begin(), uniqueNumbers.end());
                    basemap_group.insert(basemap_group_i);
                }
                all_basemap_group.push_back(basemap_group);

                // 单时相底图方法测试
                if (max_basemap_num_i == 1)
                    break;


                //// 打印当前max_basemap_num_i下所有唯一的随机数组
                //cout << "Random groups for max_basemap_num_i = " << max_basemap_num_i << ":" << endl;
                //for (const auto& group : basemap_group) {
                //    for (int num : group) {
                //        cout << num << " ";
                //    }
                //    cout << endl;
                //}
                //cout << "-----" << endl;
            }


            vector<vector<double>> finalConfidenceMatrix;
        
            for (int error_i = 0; error_i < Errors.size(); error_i++)
            {
                vector <double> h_RMSE;
                vector <double> v_RMSE;
                vector <double> RMSE;
                for(int max_basemap_num_i = 0;max_basemap_num_i<max_basemap_num;max_basemap_num_i++)
                {
                    double h_RMSE_mean = 0;
                    double v_RMSE_mean = 0;
                    double RMSE_mean = 0;
                    for (int basemap_random_times_i = 0; basemap_random_times_i < basemap_random_times; basemap_random_times_i++)
                    {
                        vector <vector <Point3d>> BM_All_Points_GE_All;
                        vector <vector <Point3d>> BM_All_Points_DSM_GE_All;

                        //这个for循环的目的是将所有分块区域的匹配点都加入到同一个vector数组中，从而可以达到和整体平差基本一样的结果
                        for (int folder_area = 0; folder_area < Chunked_Folder.size(); folder_area++)
                        {
                            if (folder_area == 22)
                            {
                                int debug = 1;
                            }

                            // 2、修改读取点位置的函数，传入三个方向的误差，进行添加误差处理
                            vector <vector <Point3d>> BM_All_Points_GE;
                            vector <vector <Point3d>> BM_All_Points_DSM_GE;
                            GetAllFiles_txt2GeoInfo_BM_and_DSM_Para_Adjust_Basemap_Num(All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[folder_area], "GE", BM_All_Points_GE, BM_All_Points_DSM_GE, Errors[error_i].x, Errors[error_i].y, Errors[error_i].z, all_basemap_group[max_basemap_num_i], basemap_random_times_i);
                            for (int i = 0; i < BM_All_Points_GE.size(); i++)
                                BM_All_Points_GE_All.push_back(BM_All_Points_GE[i]);
                            for (int i = 0; i < BM_All_Points_DSM_GE.size(); i++)
                                BM_All_Points_DSM_GE_All.push_back(BM_All_Points_DSM_GE[i]);
                        }

                        Point3d Model_Translation_Parameter_GE_Ransac;
                        Point3d Model_Translation_Parameter_GE_Iter;
                        string Mask_Prob_Path = All_Area_Dir + "\\" + Area_Folder[0] + "\\Building_Mask_Prob";

                        vector <vector<float>> Building_Prob_Weight;
                        // 遍历BM_All_Points_GE_All来定义Building_Prob_Weight的结构
                        for (const auto& subVector : BM_All_Points_GE_All) {
                            // 使用subVector的大小来创建一个float向量
                            Building_Prob_Weight.push_back(std::vector<float>(subVector.size(), 0.0f));  // 初始化为0.0f
                        }

                        RANSAC_Point_Semantic_Constraint(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All, Building_Prob_Weight, 6000, RANSAC_THRESHOLD, 0.995, RANSAC_RATIO, Model_Translation_Parameter_GE_Ransac, string("GE"), All_Area_Dir + "\\" + Area_Folder[0], Mask_Prob_Path, Mask_Type, "GE", Cut_Num);

                        // RANSAC_Point(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All, 6000, RANSAC_THRESHOLD, 0.995, RANSAC_RATIO, Model_Translation_Parameter_GE_Ransac, string("GE"), All_Area_Dir + "\\" + Area_Folder[0]);

                        double x_GE = Model_Translation_Parameter_GE_Ransac.x;
                        double y_GE = Model_Translation_Parameter_GE_Ransac.y;
                        double z_GE = Model_Translation_Parameter_GE_Ransac.z;

                        if (is_Building_Mask == true)
                        {
                            // printf("正在进行基于%s建筑物掩膜的粗差剔除......\n", Mask_Type.c_str());
                            string Mask_Path = All_Area_Dir + "\\" + Area_Folder[0] + "\\Building_Mask";
                            Building_Mask_Elimination_Cross_Error(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All, Mask_Path, Mask_Type, "GE", Cut_Num);
                            // printf("已完成基于%s建筑物掩膜的粗差剔除\n", Mask_Type.c_str());
                        }

                        // 计算高程的权重（置信度）
                        string Mask_Path = All_Area_Dir + "\\" + Area_Folder[0] + "\\Building_Mask";

                        Elevation_Weight_Calculation_Basemap_Num(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All, Mask_Path, "GE", All_Area_Dir + "\\" + Area_Folder[0], finalConfidenceMatrix, all_basemap_group[max_basemap_num_i], basemap_random_times_i);
                        // printf("已完成高程赋权\n");

                        Weight_Iteration_Point_Elevation(BM_All_Points_GE_All, BM_All_Points_DSM_GE_All, x_GE, y_GE, z_GE, WEIGHT_ITERATION_SIGMA_BEF, Model_Translation_Parameter_GE_Iter, finalConfidenceMatrix, All_Area_Dir + "\\" + Area_Folder[0]);

                        double x_GE_iter = Model_Translation_Parameter_GE_Iter.x;
                        double y_GE_iter = Model_Translation_Parameter_GE_Iter.y;
                        double z_GE_iter = Model_Translation_Parameter_GE_Iter.z;

                        // 单时相底图评估
                        Point3d Model_Params;

                        Model_Params.x = x_GE_iter;
                        Model_Params.y = y_GE_iter;
                        Model_Params.z = z_GE_iter;

                        calculateGeoPositioningAccuracy("Coarse+Fine Correction", Model_Params, All_Area_Dir + "\\Truth_GCP.txt", All_Area_Dir + "\\Unrectified_GCP.txt");

                        // printf("已完成GE的精校正\n");
                        double h_RMSE_tmp = sqrt(pow(abs(Errors[error_i].x + x_GE_iter), 2) + pow(abs(Errors[error_i].y + y_GE_iter), 2));
                        double v_RMSE_tmp = abs(Errors[error_i].z + z_GE_iter);
                        double RMSE_tmp = sqrt(pow(h_RMSE_tmp, 2)+pow(v_RMSE_tmp,2));
                        printf("\n model parameters: x_iter %lf y_iter %lf z_iter %lf \n", x_GE_iter, y_GE_iter, z_GE_iter);
                        printf("x %lf y %lf z %lf delta_x %lf delta_y %lf h_RMSE %lf v_RMSE %lf RMSE %lf Basemap_Num %d\n", Errors[error_i].x, Errors[error_i].y, Errors[error_i].z, abs(Errors[error_i].x + x_GE_iter), abs(Errors[error_i].y + y_GE_iter), h_RMSE_tmp, v_RMSE_tmp, RMSE_tmp, max_basemap_num_i+1);
                        
                        h_RMSE_mean += h_RMSE_tmp;
                        v_RMSE_mean += v_RMSE_tmp;
                        RMSE_mean += RMSE_tmp;
                        
                        if (basemap_random_times_i == basemap_random_times - 1)
                        {
                            h_RMSE_mean /= basemap_random_times;
                            v_RMSE_mean /= basemap_random_times;
                            RMSE_mean /= basemap_random_times;

                            h_RMSE.push_back(h_RMSE_mean);
                            v_RMSE.push_back(v_RMSE_mean);
                            RMSE.push_back(RMSE_mean);
                        }
                    }
                }
                // 4、计算校正参数，和添加的误差进行对比，从而得到校正误差，计算水平方向RMSE、垂直方向RMSE以及总的RMSE

                FILE* rmse1;
                rmse1 = fopen((All_Area_Dir + "\\" + "RMSE_Results_Basemap_Num.txt").c_str(), "w");
                // 计算RMSE
                for (int ii = 0; ii < RMSE.size(); ii++)//10
                {
                    fprintf(rmse1, "x %lf y %lf z %lf h_RMSE %lf v_RMSE %lf RMSE %lf Basemap_Num %d\n", Errors[error_i].x, Errors[error_i].y, Errors[error_i].z, h_RMSE[ii], v_RMSE[ii], RMSE[ii], ii + 1);
                }

                fclose(rmse1); // 关闭文件
            }


        }

    }

}


void GetAllFiles_txt2GeoInfo_BM_and_DSM_Para_Adjust(string dir_path, string BM_type, vector <vector <Point3d>>& BM_All_Points_input, vector <vector <Point3d>>& DSM_All_Points_input, double error_x, double error_y, double error_z)
{
    _int64 hFile = 0;
    //文件信息结构体
    vector<string> txt_files_BM;
    vector<string> txt_files_DSM;
    string path = dir_path + "\\" + "MATCH_RESULT" + "\\" + BM_type;
    struct _finddata_t fileinfo;
    string p;
    if ((hFile = _findfirst(p.assign(path).append("\\*kpt1.txt").c_str(), &fileinfo)) != -1)
    {
        do
        {
            txt_files_BM.push_back(fileinfo.name);
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }

    if ((hFile = _findfirst(p.assign(path).append("\\*kpt0.txt").c_str(), &fileinfo)) != -1)
    {
        do
        {
            txt_files_DSM.push_back(fileinfo.name);
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }

    // 底图文件读取初始化
    vector <string> tif_files;
    vector <string> tif_dem;
    string BM_tif_path = dir_path + "\\" + "BASEMAP" + "\\" + BM_type;
    string BM_dem_path = dir_path + "\\" + "BASEMAP" + "\\" + "SRTM";
    GetAllFiles_tif(BM_tif_path, tif_files);
    GetAllFiles_tif(BM_dem_path, tif_dem);
    vector <vector <Point3d>> BM_All_Points(txt_files_BM.size());

    // DSM、DOM文件读取初始化
    vector <string> tif_dom;
    vector <string> tif_dsm;
    vector <string> dom_tfw;
    vector <string> dsm_tfw;
    string DOM_path = dir_path + "\\" + "DOM";
    string DSM_path = dir_path + "\\" + "DSM";
    GetAllFiles_tif(DOM_path, tif_dom);
    GetAllFiles_tif(DSM_path, tif_dsm);
    GetAllFiles_tfw(DOM_path, dom_tfw);
    GetAllFiles_tfw(DSM_path, dsm_tfw);
    vector <vector <Point3d>> DSM_All_Points(txt_files_DSM.size());

    GDALAllRegister();//注册所有的驱动
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名

    // 开始循环
    for (int i = 0; i < txt_files_BM.size(); i++)
    {
        // BASEMAP
        Point3d temp_pt_BM;
        FILE* p_BM;
        string path_txt_BM = path + "\\" + txt_files_BM[i];
        p_BM = fopen(path_txt_BM.c_str(), "r");


        string path_tif_BM = BM_tif_path + "\\" + tif_files[i];
        //加载tif数据
        GDALDataset* poDataset_BM = (GDALDataset*)GDALOpen(path_tif_BM.c_str(), GA_ReadOnly);//GA_Update和GA_ReadOnly两种模式
        if (poDataset_BM == NULL) {
            std::cout << "指定的文件不能打开!" << std::endl;
            exit(0);
        }

        //获取坐标变换系数
        double transBM[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
        poDataset_BM->GetGeoTransform(transBM);//影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。

        string path_dem = BM_dem_path + "\\" + tif_dem[0];
        GDALDataset* poDataset_DEM = (GDALDataset*)GDALOpen(path_dem.c_str(), GA_ReadOnly);
        if (poDataset_DEM == NULL) {
            std::cout << "指定的文件不能打开!" << std::endl;
            exit(0);
        }

        //获取dem坐标变换系数
        double transdem[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
        poDataset_DEM->GetGeoTransform(transdem);//影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。
        transdem[0] += 1.0 / 2 * transdem[1];
        transdem[3] += 1.0 / 2 * transdem[5];
        double x_BM, y_BM;



        //DOM+DSM
        Point3d temp_pt_DOM;
        FILE* p_DOM;
        string path_txt_DOM = path + "\\" + txt_files_DSM[i];
        p_DOM = fopen(path_txt_DOM.c_str(), "r");


        string path_tif_DOM = DOM_path + "\\" + tif_dom[0];
        //加载tif数据
        GDALDataset* poDataset_DOM = (GDALDataset*)GDALOpen(path_tif_DOM.c_str(), GA_ReadOnly);//GA_Update和GA_ReadOnly两种模式
        if (poDataset_DOM == NULL) {
            std::cout << "指定的文件不能打开!" << std::endl;
            exit(0);
        }

        //获取dom坐标变换系数
        double trans_DOM[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
        poDataset_DOM->GetGeoTransform(trans_DOM);//影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。

        string path_dsm = DSM_path + "\\" + tif_dsm[0];
        GDALDataset* poDataset_DSM = (GDALDataset*)GDALOpen(path_dsm.c_str(), GA_ReadOnly);//GA_Update和GA_ReadOnly两种模式
        if (poDataset_DSM == NULL) {
            std::cout << "指定的文件不能打开!" << std::endl;
            exit(0);
        }

        //获取dem坐标变换系数
        double transdsm[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
        poDataset_DSM->GetGeoTransform(transdsm);//影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。

        double x_DOM, y_DOM;



        // 开始读取txt文件
        while (fscanf(p_BM, "%lf %lf", &x_BM, &y_BM) != EOF && fscanf(p_DOM, "%lf %lf", &x_DOM, &y_DOM) != EOF)
        {

            //计算图像地理坐标,若图像中某一点的行数和列数分别为：row, column,则该点的地理坐标为：
            temp_pt_BM.x = transBM[0] + x_BM * transBM[1] + y_BM * transBM[2];
            temp_pt_BM.y = transBM[3] + x_BM * transBM[4] + y_BM * transBM[5];

            //计算行列号
            double dTemp_BM = transdem[2] * transdem[4] - transdem[1] * transdem[5];
            double dRow_BM = (transdem[4] * (temp_pt_BM.x - transdem[0]) - transdem[1] * (temp_pt_BM.y - transdem[3])) / dTemp_BM;
            double dCol_BM = (transdem[2] * (temp_pt_BM.y - transdem[3]) - transdem[5] * (temp_pt_BM.x - transdem[0])) / dTemp_BM;
            int dx_BM = int(dCol_BM + 0);
            int dy_BM = int(dRow_BM + 0);//这里出问题

            //计算图像地理坐标,若图像中某一点的行数和列数分别为：row, column,则该点的地理坐标为：
            temp_pt_DOM.x = trans_DOM[0] + x_DOM * trans_DOM[1] + y_DOM * trans_DOM[2] + error_x;
            temp_pt_DOM.y = trans_DOM[3] + x_DOM * trans_DOM[4] + y_DOM * trans_DOM[5] + error_y;

            //计算行列号
            double dTemp_DOM = transdsm[2] * transdsm[4] - transdsm[1] * transdsm[5];
            double dRow_DOM = (transdsm[4] * (temp_pt_DOM.x - transdsm[0]) - transdsm[1] * (temp_pt_DOM.y - transdsm[3])) / dTemp_DOM;
            double dCol_DOM = (transdsm[2] * (temp_pt_DOM.y - transdsm[3]) - transdsm[5] * (temp_pt_DOM.x - transdsm[0])) / dTemp_DOM;
            int dx_DOM = int(dCol_DOM + 0);
            int dy_DOM = int(dRow_DOM + 0);

            // 一般高程数据类型就是float，因此不再进行判断，如果为了严谨应该判断类型
            GDALRasterBand* poBand_DEM = poDataset_DEM->GetRasterBand(1);
            float* buffer_DEM = (float*)CPLMalloc(sizeof(float) * 1);
            if (dx_BM < 0 || dy_BM < 0 || dx_BM >= poDataset_DEM->GetRasterXSize() || dy_BM >= poDataset_DEM->GetRasterXSize())
            {
                continue;
            }

            GDALRasterBand* poBand_DSM = poDataset_DSM->GetRasterBand(1);
            float* buffer_DSM = (float*)CPLMalloc(sizeof(float) * 1);
            if (dx_DOM < 0 || dy_DOM < 0 || dx_DOM >= poDataset_DSM->GetRasterXSize() || dy_DOM >= poDataset_DSM->GetRasterXSize())
            {
                continue;
            }
            if (GDALRasterIO(poBand_DEM, GF_Read, dx_BM, dy_BM, 1, 1, buffer_DEM, 1, 1, GDT_Float32, 0, 0) != CE_None) exit(0);
            if (GDALRasterIO(poBand_DSM, GF_Read, dx_DOM, dy_DOM, 1, 1, buffer_DSM, 1, 1, GDT_Float32, 0, 0) != CE_None) exit(0);

            // 赋值像素高程
            temp_pt_BM.z = double(*buffer_DEM);
            BM_All_Points[i].push_back(temp_pt_BM);
            CPLFree(buffer_DEM);

            // 赋值像素高程
            temp_pt_DOM.z = double(*buffer_DSM) + error_z;
            DSM_All_Points[i].push_back(temp_pt_DOM);
            CPLFree(buffer_DSM);
        }
        GDALClose(poDataset_BM);
        GDALClose(poDataset_DEM);
        GDALClose(poDataset_DOM);
        GDALClose(poDataset_DSM);
        fclose(p_BM);
        fclose(p_DOM);
    }
    BM_All_Points_input = move(BM_All_Points);
    DSM_All_Points_input = move(DSM_All_Points);
}


void GetAllFiles_txt2GeoInfo_BM_and_DSM_Para_Adjust_Basemap_Num(string dir_path, string BM_type, vector <vector <Point3d>>& BM_All_Points_input, vector <vector <Point3d>>& DSM_All_Points_input, double error_x, double error_y, double error_z, set <vector<int>> basemap_group, int basemap_random_times_i)
{
    // 调用函数得到当前的底图组合
    vector<int> basemap_group_now = getVectorFromSet(basemap_group, basemap_random_times_i);

    _int64 hFile = 0;
    //文件信息结构体
    vector<string> txt_files_BM;
    vector<string> txt_files_DSM;
    string path = dir_path + "\\" + "MATCH_RESULT" + "\\" + BM_type;
    struct _finddata_t fileinfo;
    string p;
    if ((hFile = _findfirst(p.assign(path).append("\\*kpt1.txt").c_str(), &fileinfo)) != -1)
    {
        do
        {
            txt_files_BM.push_back(fileinfo.name);
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }

    if ((hFile = _findfirst(p.assign(path).append("\\*kpt0.txt").c_str(), &fileinfo)) != -1)
    {
        do
        {
            txt_files_DSM.push_back(fileinfo.name);
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }

    // 底图文件读取初始化
    vector <string> tif_files;
    vector <string> tif_dem;
    string BM_tif_path = dir_path + "\\" + "BASEMAP" + "\\" + BM_type;
    string BM_dem_path = dir_path + "\\" + "BASEMAP" + "\\" + "SRTM";
    GetAllFiles_tif(BM_tif_path, tif_files);
    GetAllFiles_tif(BM_dem_path, tif_dem);
    vector <vector <Point3d>> BM_All_Points(basemap_group_now.size());

    // DSM、DOM文件读取初始化
    vector <string> tif_dom;
    vector <string> tif_dsm;
    vector <string> dom_tfw;
    vector <string> dsm_tfw;
    string DOM_path = dir_path + "\\" + "DOM";
    string DSM_path = dir_path + "\\" + "DSM";
    GetAllFiles_tif(DOM_path, tif_dom);
    GetAllFiles_tif(DSM_path, tif_dsm);
    GetAllFiles_tfw(DOM_path, dom_tfw);
    GetAllFiles_tfw(DSM_path, dsm_tfw);
    vector <vector <Point3d>> DSM_All_Points(basemap_group_now.size());



    GDALAllRegister();//注册所有的驱动
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名

    // 开始循环
    for (int i = 0; i < basemap_group_now.size(); i++)
    {
        // BASEMAP
        Point3d temp_pt_BM;
        FILE* p_BM;
        string path_txt_BM = path + "\\" + txt_files_BM[basemap_group_now[i]];
        p_BM = fopen(path_txt_BM.c_str(), "r");


        string path_tif_BM = BM_tif_path + "\\" + tif_files[basemap_group_now[i]];
        //加载tif数据
        GDALDataset* poDataset_BM = (GDALDataset*)GDALOpen(path_tif_BM.c_str(), GA_ReadOnly);//GA_Update和GA_ReadOnly两种模式
        if (poDataset_BM == NULL) {
            std::cout << "指定的文件不能打开!" << std::endl;
            exit(0);
        }

        //获取坐标变换系数
        double transBM[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
        poDataset_BM->GetGeoTransform(transBM);//影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。

        string path_dem = BM_dem_path + "\\" + tif_dem[0];
        GDALDataset* poDataset_DEM = (GDALDataset*)GDALOpen(path_dem.c_str(), GA_ReadOnly);
        if (poDataset_DEM == NULL) {
            std::cout << "指定的文件不能打开!" << std::endl;
            exit(0);
        }

        //获取dem坐标变换系数
        double transdem[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
        poDataset_DEM->GetGeoTransform(transdem);//影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。
        transdem[0] += 1.0 / 2 * transdem[1];
        transdem[3] += 1.0 / 2 * transdem[5];
        double x_BM, y_BM;



        //DOM+DSM
        Point3d temp_pt_DOM;
        FILE* p_DOM;
        string path_txt_DOM = path + "\\" + txt_files_DSM[basemap_group_now[i]];
        p_DOM = fopen(path_txt_DOM.c_str(), "r");


        string path_tif_DOM = DOM_path + "\\" + tif_dom[0];
        //加载tif数据
        GDALDataset* poDataset_DOM = (GDALDataset*)GDALOpen(path_tif_DOM.c_str(), GA_ReadOnly);//GA_Update和GA_ReadOnly两种模式
        if (poDataset_DOM == NULL) {
            std::cout << "指定的文件不能打开!" << std::endl;
            exit(0);
        }

        //获取dom坐标变换系数
        double trans_DOM[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
        poDataset_DOM->GetGeoTransform(trans_DOM);//影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。

        string path_dsm = DSM_path + "\\" + tif_dsm[0];
        GDALDataset* poDataset_DSM = (GDALDataset*)GDALOpen(path_dsm.c_str(), GA_ReadOnly);//GA_Update和GA_ReadOnly两种模式
        if (poDataset_DSM == NULL) {
            std::cout << "指定的文件不能打开!" << std::endl;
            exit(0);
        }

        //获取dem坐标变换系数
        double transdsm[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
        poDataset_DSM->GetGeoTransform(transdsm);//影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。

        double x_DOM, y_DOM;



        // 开始读取txt文件
        while (fscanf(p_BM, "%lf %lf", &x_BM, &y_BM) != EOF && fscanf(p_DOM, "%lf %lf", &x_DOM, &y_DOM) != EOF)
        {

            //计算图像地理坐标,若图像中某一点的行数和列数分别为：row, column,则该点的地理坐标为：
            temp_pt_BM.x = transBM[0] + x_BM * transBM[1] + y_BM * transBM[2];
            temp_pt_BM.y = transBM[3] + x_BM * transBM[4] + y_BM * transBM[5];

            //计算行列号
            double dTemp_BM = transdem[2] * transdem[4] - transdem[1] * transdem[5];
            double dRow_BM = (transdem[4] * (temp_pt_BM.x - transdem[0]) - transdem[1] * (temp_pt_BM.y - transdem[3])) / dTemp_BM;
            double dCol_BM = (transdem[2] * (temp_pt_BM.y - transdem[3]) - transdem[5] * (temp_pt_BM.x - transdem[0])) / dTemp_BM;
            int dx_BM = int(dCol_BM + 0);
            int dy_BM = int(dRow_BM + 0);//这里出问题

            //计算图像地理坐标,若图像中某一点的行数和列数分别为：row, column,则该点的地理坐标为：
            temp_pt_DOM.x = trans_DOM[0] + x_DOM * trans_DOM[1] + y_DOM * trans_DOM[2] + error_x;
            temp_pt_DOM.y = trans_DOM[3] + x_DOM * trans_DOM[4] + y_DOM * trans_DOM[5] + error_y;

            //计算行列号
            double dTemp_DOM = transdsm[2] * transdsm[4] - transdsm[1] * transdsm[5];
            double dRow_DOM = (transdsm[4] * (temp_pt_DOM.x - transdsm[0]) - transdsm[1] * (temp_pt_DOM.y - transdsm[3])) / dTemp_DOM;
            double dCol_DOM = (transdsm[2] * (temp_pt_DOM.y - transdsm[3]) - transdsm[5] * (temp_pt_DOM.x - transdsm[0])) / dTemp_DOM;
            int dx_DOM = int(dCol_DOM + 0);
            int dy_DOM = int(dRow_DOM + 0);

            // 一般高程数据类型就是float，因此不再进行判断，如果为了严谨应该判断类型
            GDALRasterBand* poBand_DEM = poDataset_DEM->GetRasterBand(1);
            float* buffer_DEM = (float*)CPLMalloc(sizeof(float) * 1);
            if (dx_BM < 0 || dy_BM < 0 || dx_BM >= poDataset_DEM->GetRasterXSize() || dy_BM >= poDataset_DEM->GetRasterXSize())
            {
                continue;
            }

            GDALRasterBand* poBand_DSM = poDataset_DSM->GetRasterBand(1);
            float* buffer_DSM = (float*)CPLMalloc(sizeof(float) * 1);
            if (dx_DOM < 0 || dy_DOM < 0 || dx_DOM >= poDataset_DSM->GetRasterXSize() || dy_DOM >= poDataset_DSM->GetRasterXSize())
            {
                continue;
            }
            if (GDALRasterIO(poBand_DEM, GF_Read, dx_BM, dy_BM, 1, 1, buffer_DEM, 1, 1, GDT_Float32, 0, 0) != CE_None) exit(0);
            if (GDALRasterIO(poBand_DSM, GF_Read, dx_DOM, dy_DOM, 1, 1, buffer_DSM, 1, 1, GDT_Float32, 0, 0) != CE_None) exit(0);

            // 赋值像素高程
            temp_pt_BM.z = double(*buffer_DEM);
            BM_All_Points[i].push_back(temp_pt_BM);
            CPLFree(buffer_DEM);

            // 赋值像素高程
            temp_pt_DOM.z = double(*buffer_DSM) + error_z;
            DSM_All_Points[i].push_back(temp_pt_DOM);
            CPLFree(buffer_DSM);
        }
        GDALClose(poDataset_BM);
        GDALClose(poDataset_DEM);
        GDALClose(poDataset_DOM);
        GDALClose(poDataset_DSM);
        fclose(p_BM);
        fclose(p_DOM);
    }
    BM_All_Points_input = move(BM_All_Points);
    DSM_All_Points_input = move(DSM_All_Points);
}


// 函数：从set中获取第n个vector
vector<int> getVectorFromSet(const set<vector<int>>& basemap_group, int n) {
    auto it = basemap_group.begin(); // 获取set的开始迭代器
    // 使用std::advance安全地移动迭代器，但注意这是线性时间操作
    advance(it, n);
    return (it != basemap_group.end()) ? *it : vector<int>(); // 如果迭代器没有超出set的末尾，则返回找到的vector，否则返回空vector
}


void Elevation_Weight_Calculation_Basemap_Num(vector<vector<Point3d>>& BM_pt, vector<vector<Point3d>>& DSM_pt, string Mask_Path, string BM_Type, string dir_path, vector<vector<double>>& finalConfidenceMatrix, set<vector<int>> basemap_group, int basemap_random_times_i) {
    // 调用函数得到当前的底图组合
    vector<int> basemap_group_now = getVectorFromSet(basemap_group, basemap_random_times_i);


    vector<string> tif_dem;
    string BM_dem_path = dir_path + "\\" + "BASEMAP" + "\\" + "SRTM";
    GetAllFiles_tif(BM_dem_path, tif_dem);

    vector<string> files;
    Mask_Path = Mask_Path + "\\" + BM_Type;
    GetAllFiles_tif(Mask_Path, files);

    if (files.size() == 0) {
        printf("无掩膜文件，请检查输入文件夹！\n");
        return;
    }

    GDALAllRegister();
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名

    // 读取SRTM数据
    GDALDataset* poDataset = (GDALDataset*)GDALOpen((BM_dem_path + "\\" + tif_dem[0]).c_str(), GA_ReadOnly);
    if (poDataset == NULL) {
        cout << "无法读取SRTM数据！" << endl;
        return;
    }

    int nXSize = poDataset->GetRasterXSize();
    int nYSize = poDataset->GetRasterYSize();
    double adfGeoTransform[6];
    poDataset->GetGeoTransform(adfGeoTransform);
    double cellSizeX = adfGeoTransform[1]; // 一个格子的宽度
    double cellSizeY = -adfGeoTransform[5]; // 一个格子的高度（通常为负值）


    //vector<vector<double>> sharedConfidenceMatrix(nYSize, vector<double>(nXSize, 0.0));

//#pragma omp parallel
//    {
//        vector<vector<double>> localConfidenceMatrix(nYSize, vector<double>(nXSize, 0.0));
//
//#pragma omp for nowait
//        for (int fileIndex = 0; fileIndex < files.size(); ++fileIndex) {
//            const string& file = files[fileIndex];
//            GDALDataset* poMaskDS = (GDALDataset*)GDALOpen((Mask_Path + "\\" + file).c_str(), GA_ReadOnly);
//            if (poMaskDS == NULL) {
//                cout << "无法读取掩膜文件：" << file << endl;
//                continue;
//            }
//
//            double maskGeoTransform[6];
//            if (poMaskDS->GetGeoTransform(maskGeoTransform) != CE_None) {
//                GDALClose(poMaskDS);
//                continue;
//            }
//
//            double maskMinX = maskGeoTransform[0];
//            double maskMaxX = maskGeoTransform[0] + poMaskDS->GetRasterXSize() * maskGeoTransform[1];
//            double maskMaxY = maskGeoTransform[3];
//            double maskMinY = maskGeoTransform[3] + poMaskDS->GetRasterYSize() * maskGeoTransform[5];
//
//            for (int i = 0; i < nYSize; ++i) {
//                for (int j = 0; j < nXSize; ++j) {
//                    double cellGeoX = adfGeoTransform[0] + j * cellSizeX;
//                    double cellGeoY = adfGeoTransform[3] - i * cellSizeY;
//
//                    if (cellGeoX < maskMaxX && cellGeoX + cellSizeX > maskMinX &&
//                        cellGeoY < maskMaxY && cellGeoY - cellSizeY > maskMinY) {
//                        localConfidenceMatrix[i][j] += CalculateConfidenceForCell(poMaskDS, cellGeoX, cellGeoY, cellSizeX, cellSizeY);
//                    }
//                    else {
//                        localConfidenceMatrix[i][j] += 1.0; // 赋予默认置信度
//                    }
//                }
//            }
//            GDALClose(poMaskDS);
//        }
//
//        // 将局部矩阵累加到共享矩阵中
//#pragma omp critical
//        for (int i = 0; i < nYSize; ++i) {
//            for (int j = 0; j < nXSize; ++j) {
//                sharedConfidenceMatrix[i][j] += localConfidenceMatrix[i][j];
//            }
//        }
//    }


    // 存储每张掩膜的置信度矩阵
    // 为外层向量分配大小，每个元素是一个vector<double>
    finalConfidenceMatrix.resize(nYSize);

    // 为每个内层向量（即每一行）分配大小和初始值
    for (int i = 0; i < nYSize; ++i) {
        finalConfidenceMatrix[i].resize(nXSize, 0.0);
    }

    for (int file_i = 0; file_i < basemap_group_now.size();file_i++) {
        GDALAllRegister();
        CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");  //设置支持中文路径和文件名

        GDALDataset* poMaskDS = (GDALDataset*)GDALOpen((Mask_Path + "\\" + files[basemap_group_now[file_i]]).c_str(), GA_ReadOnly);
        if (poMaskDS == NULL) {
            cout << "无法读取掩膜文件：" << files[basemap_group_now[file_i]] << endl;
            continue;
        }

        // 获取掩膜文件的地理转换参数
        double maskGeoTransform[6];
        if (poMaskDS->GetGeoTransform(maskGeoTransform) != CE_None) {
            // 错误处理
            GDALClose(poMaskDS);
            continue;
        }

        // 获取掩膜文件的空间范围
        double maskMinX = maskGeoTransform[0];
        double maskMaxX = maskGeoTransform[0] + poMaskDS->GetRasterXSize() * maskGeoTransform[1];
        double maskMaxY = maskGeoTransform[3];
        double maskMinY = maskGeoTransform[3] + poMaskDS->GetRasterYSize() * maskGeoTransform[5]; // 通常为负值

        // 计算当前掩膜的置信度
        for (int i = 0; i < nYSize; ++i) {
            // assert(i < finalConfidenceMatrix.size());
            for (int j = 0; j < nXSize; ++j) {
                // assert(j < finalConfidenceMatrix[i].size());
                double cellGeoX = adfGeoTransform[0] + j * cellSizeX;
                double cellGeoY = adfGeoTransform[3] - i * cellSizeY;

                // 检查SRTM方格是否在掩膜范围内
                if (cellGeoX < maskMaxX && cellGeoX + cellSizeX > maskMinX &&
                    cellGeoY < maskMaxY && cellGeoY - cellSizeY > maskMinY) {
                    finalConfidenceMatrix[i][j] += CalculateConfidenceForCell(poMaskDS, cellGeoX, cellGeoY, cellSizeX, cellSizeY);
                }
                else {
                    // 方格不在掩膜范围内，可以选择跳过或赋予默认置信度
                    finalConfidenceMatrix[i][j] += 1.0; // 例如，赋予默认置信度1.0
                }
            }
        }

        GDALClose(poMaskDS);

    }


    // 计算每个格子的平均置信度
    // 确保 finalConfidenceMatrix 的尺寸正确
    // assert(finalConfidenceMatrix.size() == nYSize);
    //for (const auto& row : finalConfidenceMatrix) {
    //    assert(row.size() == nXSize);
    //}
    for (auto& row : finalConfidenceMatrix) {
        for (double& value : row) {
            value /= basemap_group_now.size();
        }
    }

    //// 将最终置信度应用到BM_pt
    //// // 初始化 Elevation_Weight 矩阵，使其与 BM_pt 一样大小，并填充为0
    //Elevation_Weight.resize(BM_pt.size());
    //for (size_t i = 0; i < BM_pt.size(); ++i) {
    //    Elevation_Weight[i].resize(BM_pt[i].size(), 0.0);
    //}
    //// 在应用置信度到 BM_pt 之前加上断言
    ////assert(Elevation_Weight.size() == BM_pt.size());
    //for (size_t i = 0; i < BM_pt.size(); ++i) {
    //    for (size_t j = 0; j < BM_pt[i].size(); ++j) {
    //        int xIndex = static_cast<int>((BM_pt[i][j].x - adfGeoTransform[0]) / cellSizeX);
    //        int yIndex = static_cast<int>((BM_pt[i][j].y - adfGeoTransform[3]) / -cellSizeY);
    //        if (finalConfidenceMatrix[yIndex][xIndex] > 0.8)
    //        {
    //            Elevation_Weight[i][j] = finalConfidenceMatrix[yIndex][xIndex];
    //        }
    //    }
    //}
    GDALDestroyDriverManager();
}


void Image_Chunking_Process_2017(string All_Area_Dir, int Cut_Num, string exe_path, bool is_Feature_Matching, bool is_Geometric_Correction, bool is_Precision_Evaluation)
{
    //第一步，将分块后的影像放入正确的文件夹中
    vector <string> Area_Folder;
    GetFolder(All_Area_Dir, Area_Folder);
    //if (Area_Folder.size() == 1)//需要从头开始，即把Guangzhou_is_Chunked去掉
    //{
    printf("正在进行分块操作\n");
    string path_GE = All_Area_Dir + "\\" + Area_Folder[0] + "\\BASEMAP" + "\\GE";
    string path_SRTM = All_Area_Dir + "\\" + Area_Folder[0] + "\\BASEMAP" + "\\SRTM";
    if (0 != _access((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked").c_str(), 0))
    {
        // if this folder not exist, create a new one.
        _mkdir((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked").c_str());   // 返回 0 表示创建成功，-1 表示失败
    }
    Image_Chunking(path_GE, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "GE");
    Image_Chunking(path_SRTM, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "SRTM");// SRTM 不做裁剪
    //分块DOM和DSM，并把DOM放到对应的底图文件夹中
    string path_DOM_DSM = All_Area_Dir + "\\" + Area_Folder[0];
    Image_Chunking(path_DOM_DSM, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "DOM_DSM");

    printf("分块操作已完成\n");
    //}
    //else
    //{
    //    printf("无法完成底图分块操作，由于有一个以上文件夹处于数据所在路径下，默认已完成分块进行后续操作\n");
    //}

    //第二步，将底图BASEMAP文件夹中的两个文件夹中的文件写成分别命名的txt
    vector <string> Chunked_Folder;
    GetFolder(All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Chunked_Folder);
    if (is_Feature_Matching == true)
    {
        printf("正在进行特征匹配\n");
        for (int i = 0; i < Chunked_Folder.size(); i++)
        {
            vector <string> files1;
            string dir_path1 = All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\ARC_BING_MB";
            GetAllFiles_tif(dir_path1, files1);
            FILE* p1;
            p1 = fopen((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\ARC_BING_MB.txt").c_str(), "w");
            for (int j = 0; j < files1.size(); j++)
            {
                if (files1[j] != "orthophoto.tif")
                {
                    fprintf(p1, "orthophoto.tif %s\n", files1[j].c_str());
                }
            }
            fclose(p1);

            vector <string> files2;
            string dir_path2 = All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\GE";
            GetAllFiles_tif(dir_path2, files2);
            FILE* p2;
            p2 = fopen((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\GE.txt").c_str(), "w");
            for (int j = 0; j < files2.size(); j++)
            {
                if (files2[j] != "orthophoto.tif")
                {
                    fprintf(p2, "orthophoto.tif %s\n", files2[j].c_str());
                }
            }
            fclose(p2);
        }

        //第三步，写bat文件，调用match_pair.exe,注意输入文件夹，输入文件的txt，输出路径
        for (int i = 0; i < Chunked_Folder.size(); i++)
        {
            FILE* p3;
            p3 = fopen((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\ARC_BING_MB_Match_Pair.bat").c_str(), "w");
            fprintf(p3, ("cmd /c \"cd /d " + exe_path + "&&" + exe_path + "/match_pairs.exe --resize -1 --superglue outdoor --max_keypoints -1 --keypoint_threshold 0.04 --nms_radius 3 --match_threshold 0.90 --resize_float --viz --viz_extension png --input_dir %s --input_pairs %s --output_dir %s\"").c_str(),
                (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\ARC_BING_MB/").c_str(), (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\ARC_BING_MB.txt").c_str(), (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\MATCH_RESULT\\ARC_BING_MB/").c_str());
            fclose(p3);
            FILE* p4;
            p4 = fopen((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\GE_Match_Pair.bat").c_str(), "w");
            fprintf(p4, ("cmd /c \"cd /d " + exe_path + "&&" + exe_path + "/match_pairs.exe --resize -1 --superglue outdoor --max_keypoints -1 --keypoint_threshold 0.04 --nms_radius 3 --match_threshold 0.90 --resize_float --viz --viz_extension png --input_dir %s --input_pairs %s --output_dir %s\"").c_str(),
                (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\GE/").c_str(), (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\GE.txt").c_str(), (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\MATCH_RESULT\\GE/").c_str());
            fclose(p4);

            system((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\ARC_BING_MB_Match_Pair.bat").c_str());
            system((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\GE_Match_Pair.bat").c_str());
        }
        printf("已完成特征匹配\n");
    }

    if (is_Geometric_Correction == true)
    {
        printf("正在进行DSM和DOM的几何校正......\n");
        //第四步，is_merged为true，改算法里面的循环；
        vector <vector <Point3d>> BM_All_Points_GE_All;
        vector <vector <Point3d>> BM_All_Points_DSM_GE_All;

        //这个for循环的目的是将所有分块区域的匹配点都加入到同一个vector数组中，从而可以达到和整体平差基本一样的结果
        for (int folder_area = 0; folder_area < Chunked_Folder.size(); folder_area++)
        {
            vector <vector <Point3d>> BM_All_Points_GE;
            vector <vector <Point3d>> BM_All_Points_DSM_GE;
            GetAllFiles_txt2GeoInfo_BM_and_DSM(All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[folder_area], "GE", BM_All_Points_GE, BM_All_Points_DSM_GE);
            for (int i = 0; i < BM_All_Points_GE.size(); i++)
                BM_All_Points_GE_All.push_back(BM_All_Points_GE[i]);
            for (int i = 0; i < BM_All_Points_DSM_GE.size(); i++)
                BM_All_Points_DSM_GE_All.push_back(BM_All_Points_DSM_GE[i]);
        }

        printf("正在进行GE的2017校正......\n");

        Point2d mean_value;
        mean_value.x = 0;
        mean_value.y = 0;
        long long n = 0;
        for (int i = 0; i < BM_All_Points_GE_All.size(); i++)
        {
            for (int j = 0; j < BM_All_Points_GE_All[i].size(); j++)
            {
                double delta_x = BM_All_Points_GE_All[i][j].x - mean_value.x;
                double delta_y = BM_All_Points_GE_All[i][j].y - mean_value.y;
                n++;
                mean_value.x += delta_x / n;
                mean_value.y += delta_y / n;
            }
        }
        printf("Point Number Before Outlier Removal: %lld\n", n);

        double variance_x = 0, variance_y = 0;
        for (int i = 0; i < BM_All_Points_GE_All.size(); i++)
        {
            for (int j = 0; j < BM_All_Points_GE_All[i].size(); j++)
            {
                double delta_x = BM_All_Points_GE_All[i][j].x - mean_value.x;
                double delta_y = BM_All_Points_GE_All[i][j].y - mean_value.y;
                variance_x += delta_x * delta_x;
                variance_y += delta_y * delta_y;
                n++;
            }
        }

        double std_x = sqrt(variance_x / n);
        double std_y = sqrt(variance_y / n);
        double threshold_x = 1.25 * std_x;
        double threshold_y = 1.25 * std_y;

        Point2d new_mean_value;
        new_mean_value.x = 0;
        new_mean_value.y = 0;
        double new_mean_value_z = 0;
        n = 0;
        for (int i = 0; i < BM_All_Points_GE_All.size(); i++)
        {
            for (int j = 0; j < BM_All_Points_GE_All[i].size(); j++)
            {
                if (fabs(BM_All_Points_GE_All[i][j].x - mean_value.x) <= threshold_x && fabs(BM_All_Points_GE_All[i][j].y - mean_value.y) <= threshold_y) 
                {
                    double delta_x = BM_All_Points_GE_All[i][j].x - new_mean_value.x;
                    double delta_y = BM_All_Points_GE_All[i][j].y - new_mean_value.y;
                    double delta_z = BM_All_Points_GE_All[i][j].z - new_mean_value_z;
                    n++;
                    new_mean_value.x += delta_x / n;
                    new_mean_value.y += delta_y / n;
                    new_mean_value_z += delta_z / n;
                }
            }
        }
        printf("Point Number After Outlier Removal: %lld\n", n);

        // 获取平移参数
        double x_GE = 0;
        double y_GE = 0;
        double z_GE = 0;
        n = 0;
        for (int i = 0; i < BM_All_Points_GE_All.size(); i++)
        {
            for (int j = 0; j < BM_All_Points_GE_All[i].size(); j++)
            {
                if (fabs(BM_All_Points_GE_All[i][j].x - mean_value.x) <= threshold_x && fabs(BM_All_Points_GE_All[i][j].y - mean_value.y) <= threshold_y)
                {
                    x_GE += new_mean_value.x - BM_All_Points_DSM_GE_All[i][j].x;
                    y_GE += new_mean_value.y - BM_All_Points_DSM_GE_All[i][j].y;
                    z_GE += new_mean_value_z - BM_All_Points_DSM_GE_All[i][j].z;
                    n++;
                }
            }
        }
        x_GE /= n;
        y_GE /= n;
        z_GE /= n;

        printf("已完成GE的2017方法校正\n");


        printf("Google_Earth_2017:\nArea_Folder     x      y     z\n");
        printf("%s %lf   %lf   %lf\n", Area_Folder[0].c_str(), x_GE, y_GE, z_GE);

        Point3d Model_Params;
        Model_Params.x = x_GE;
        Model_Params.y = y_GE;
        Model_Params.z = z_GE;

        calculateGeoPositioningAccuracy("2017", Model_Params, All_Area_Dir + "\\Truth_GCP.txt", All_Area_Dir + "\\Unrectified_GCP.txt");

        string txt_path = All_Area_Dir + "\\Model_Translate_Parameters_2017.txt";

        FILE* p_Model;
        p_Model = fopen(txt_path.c_str(), "w");

        fprintf(p_Model, "Google_Earth 2017:\nArea_Folder     x      y     z\n");
        fprintf(p_Model, "%s %lf   %lf   %lf\n", All_Area_Dir.c_str(), x_GE, y_GE, z_GE);

        fclose(p_Model);

        //第五步，根据模型参数校正DOM和DSM
        if (0 != _access((All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "Rectified_DOM_DSM_2017").c_str(), 0))
        {
            _mkdir((All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "Rectified_DOM_DSM_2017").c_str());   // 返回 0 表示创建成功，-1 表示失败
        }
        string dom_path_tif = All_Area_Dir + "\\" + Area_Folder[0] + "\\DOM\\orthophoto.tif";
        string dsm_path_tif = All_Area_Dir + "\\" + Area_Folder[0] + "\\DSM\\dsm_i.tif";

        string dom_rect_path_tif_GE_iter = All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "Rectified_DOM_DSM_2017\\orthophoto_Rect_GE.tif";
        DOM_Rectified(dom_path_tif.c_str(), dom_rect_path_tif_GE_iter.c_str(), "GTiff", x_GE, y_GE);
        string dsm_rect_path_tif_GE_iter = All_Area_Dir + "\\" + Area_Folder[0] + "\\Rectified_DOM_DSM_2017\\DSM_Rect_GE.tif";
        //实现DSM的z方向校正
        DSM_Rectified(dsm_path_tif.c_str(), dsm_rect_path_tif_GE_iter.c_str(), "GTiff", x_GE, y_GE, z_GE);
        printf("已完成DSM和DOM的几何校正\n");
    }


    //精度评价
    if (is_Precision_Evaluation == true)
    {
        printf("正在对几何校正后的DSM进行精度评价......\n");

        string TruthFile = All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "TRUTH";
        vector <string> Truth_tif;
        GetAllFiles_tif(TruthFile, Truth_tif);

        string RectFile = All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "Rectified_DOM_DSM_2017";
        vector <string> Rect_tif;
        GetAllFiles_tif(RectFile, Rect_tif);

        string UnRectFile = All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "DSM";
        vector <string> UnRect_tif;
        GetAllFiles_tif(UnRectFile, UnRect_tif);

        //去除校正后DOM的影响，因为精度评价仅需要DSM
        for (int j = 0; j < Rect_tif.size(); j++)
        {
            if (Rect_tif[j].find("DSM") == string::npos)
            {
                Rect_tif.erase(Rect_tif.begin() + j);
                j--;
            }
        }

        //创建文件夹存放精度评价矩阵
        if (0 != _access((All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation").c_str(), 0))
        {
            _mkdir((All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation").c_str());   // 返回 0 表示创建成功，-1 表示失败
        }
        if (0 != _access((All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation_UnRect").c_str(), 0))
        {
            _mkdir((All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation_UnRect").c_str());   // 返回 0 表示创建成功，-1 表示失败
        }

        //对校正后的DSM进行精度评价
        for (int j = 0; j < Rect_tif.size(); j++)
        {
            Precision_Evaluation((TruthFile + "\\" + Truth_tif[0]).c_str(), (RectFile + "\\" + Rect_tif[j]).c_str(), All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation", Area_Folder[0], false);
        }
        //对未校正的DSM进行精度评价
        Precision_Evaluation((TruthFile + "\\" + Truth_tif[0]).c_str(), (UnRectFile + "\\" + UnRect_tif[0]).c_str(), All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation_UnRect", Area_Folder[0], false);

        printf("已完成对几何校正后DSM的精度评价\n");
    }
}


// 点结构定义，包含x，y坐标，时相和序号信息
struct Point2d_kd {
    double x, y;
    double elevation;
    int time_phase;
    int index;
    bool is_chosen_1 = false;
    bool is_chosen_2 = false;
};

// 点云结构定义，适用于nanoflann的KD树实现
struct PointCloud {
    vector<Point2d_kd> pts;

    // 必需的接口函数：返回点云中点的数量
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // 必需的接口函数：返回某个点的某一维度
    inline double kdtree_get_pt(const size_t idx, int dim) const {
        return dim == 0 ? pts[idx].x : pts[idx].y;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

// 定义KD树类型，只涉及二维
typedef KDTreeSingleIndexAdaptor<
    L2_Simple_Adaptor<double, PointCloud>,
    PointCloud,
    2 /* dim */
> MyKDTree;


void Image_Chunking_Process_2024(string All_Area_Dir, int Cut_Num, string exe_path, bool is_Feature_Matching, bool is_Geometric_Correction, bool is_Precision_Evaluation)
{
    //第一步，将分块后的影像放入正确的文件夹中
    vector <string> Area_Folder;
    GetFolder(All_Area_Dir, Area_Folder);
    //if (Area_Folder.size() == 1)//需要从头开始，即把Guangzhou_is_Chunked去掉
    //{
    printf("正在进行分块操作\n");
    string path_GE = All_Area_Dir + "\\" + Area_Folder[0] + "\\BASEMAP" + "\\GE";
    string path_SRTM = All_Area_Dir + "\\" + Area_Folder[0] + "\\BASEMAP" + "\\SRTM";
    if (0 != _access((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked").c_str(), 0))
    {
        // if this folder not exist, create a new one.
        _mkdir((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked").c_str());   // 返回 0 表示创建成功，-1 表示失败
    }
    Image_Chunking(path_GE, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "GE");
    Image_Chunking(path_SRTM, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "SRTM");// SRTM 不做裁剪
    //分块DOM和DSM，并把DOM放到对应的底图文件夹中
    string path_DOM_DSM = All_Area_Dir + "\\" + Area_Folder[0];
    Image_Chunking(path_DOM_DSM, All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Cut_Num, "DOM_DSM");

    printf("分块操作已完成\n");
    //}
    //else
    //{
    //    printf("无法完成底图分块操作，由于有一个以上文件夹处于数据所在路径下，默认已完成分块进行后续操作\n");
    //}

    //第二步，将底图BASEMAP文件夹中的两个文件夹中的文件写成分别命名的txt
    vector <string> Chunked_Folder;
    GetFolder(All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked", Chunked_Folder);
    if (is_Feature_Matching == true)
    {
        printf("正在进行特征匹配\n");
        for (int i = 0; i < Chunked_Folder.size(); i++)
        {
            vector <string> files1;
            string dir_path1 = All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\ARC_BING_MB";
            GetAllFiles_tif(dir_path1, files1);
            FILE* p1;
            p1 = fopen((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\ARC_BING_MB.txt").c_str(), "w");
            for (int j = 0; j < files1.size(); j++)
            {
                if (files1[j] != "orthophoto.tif")
                {
                    fprintf(p1, "orthophoto.tif %s\n", files1[j].c_str());
                }
            }
            fclose(p1);

            vector <string> files2;
            string dir_path2 = All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\GE";
            GetAllFiles_tif(dir_path2, files2);
            FILE* p2;
            p2 = fopen((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\GE.txt").c_str(), "w");
            for (int j = 0; j < files2.size(); j++)
            {
                if (files2[j] != "orthophoto.tif")
                {
                    fprintf(p2, "orthophoto.tif %s\n", files2[j].c_str());
                }
            }
            fclose(p2);
        }

        //第三步，写bat文件，调用match_pair.exe,注意输入文件夹，输入文件的txt，输出路径
        for (int i = 0; i < Chunked_Folder.size(); i++)
        {
            FILE* p3;
            p3 = fopen((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\ARC_BING_MB_Match_Pair.bat").c_str(), "w");
            fprintf(p3, ("cmd /c \"cd /d " + exe_path + "&&" + exe_path + "/match_pairs.exe --resize -1 --superglue outdoor --max_keypoints -1 --keypoint_threshold 0.04 --nms_radius 3 --match_threshold 0.90 --resize_float --viz --viz_extension png --input_dir %s --input_pairs %s --output_dir %s\"").c_str(),
                (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\ARC_BING_MB/").c_str(), (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\ARC_BING_MB.txt").c_str(), (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\MATCH_RESULT\\ARC_BING_MB/").c_str());
            fclose(p3);
            FILE* p4;
            p4 = fopen((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\GE_Match_Pair.bat").c_str(), "w");
            fprintf(p4, ("cmd /c \"cd /d " + exe_path + "&&" + exe_path + "/match_pairs.exe --resize -1 --superglue outdoor --max_keypoints -1 --keypoint_threshold 0.04 --nms_radius 3 --match_threshold 0.90 --resize_float --viz --viz_extension png --input_dir %s --input_pairs %s --output_dir %s\"").c_str(),
                (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\GE/").c_str(), (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\BASEMAP\\GE.txt").c_str(), (All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\MATCH_RESULT\\GE/").c_str());
            fclose(p4);

            system((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\ARC_BING_MB_Match_Pair.bat").c_str());
            system((All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[i] + "\\GE_Match_Pair.bat").c_str());
        }
        printf("已完成特征匹配\n");
    }

    if (is_Geometric_Correction == true)
    {
        printf("正在进行DSM和DOM的几何校正......\n");
        //第四步，is_merged为true，改算法里面的循环；
        vector <vector <Point3d>> BM_All_Points_GE_All;
        vector <vector <Point3d>> BM_All_Points_DSM_GE_All;

        //这个for循环的目的是将所有分块区域的匹配点都加入到同一个vector数组中，从而可以达到和整体平差基本一样的结果
        for (int folder_area = 0; folder_area < Chunked_Folder.size(); folder_area++)
        {
            vector <vector <Point3d>> BM_All_Points_GE;
            vector <vector <Point3d>> BM_All_Points_DSM_GE;
            GetAllFiles_txt2GeoInfo_BM_and_DSM(All_Area_Dir + "\\" + Area_Folder[0] + "_Chunked" + "\\" + Chunked_Folder[folder_area], "GE", BM_All_Points_GE, BM_All_Points_DSM_GE);
            for (int i = 0; i < BM_All_Points_GE.size(); i++)
                BM_All_Points_GE_All.push_back(BM_All_Points_GE[i]);
            for (int i = 0; i < BM_All_Points_DSM_GE.size(); i++)
                BM_All_Points_DSM_GE_All.push_back(BM_All_Points_DSM_GE[i]);
        }

        printf("正在进行GE的2024校正......\n");

        //std::string filename = "D:\\VS_Projects\\aux_data\\BM_All_Points_GE_All_points.bin";
        //std::string filename1 = "D:\\VS_Projects\\aux_data\\BM_All_Points_DSM_GE_All_points.bin";
        //savePointsBinary(filename, BM_All_Points_GE_All);
        //savePointsBinary(filename1, BM_All_Points_DSM_GE_All);
        //return;
        //std::vector<std::vector<Point3d>> loadedPoints;
        //loadPointsBinary(filename, loadedPoints);
        
        // 排除异常值
        // 对两个数据集进行NaN值清除
        removeNaNPoints(BM_All_Points_DSM_GE_All, BM_All_Points_GE_All);

        //// DSM串点
        // 转换格式：将向量向量结构转换为一个单一的点云，用于构建KD树
        PointCloud cloud;

        for (int i = 0; i < BM_All_Points_DSM_GE_All.size(); i++)
        {
            for (int j = 0; j < BM_All_Points_DSM_GE_All[i].size(); j++)
            {
                Point2d_kd tmp_pt;
                tmp_pt.x = BM_All_Points_DSM_GE_All[i][j].x;
                tmp_pt.y = BM_All_Points_DSM_GE_All[i][j].y;
                tmp_pt.elevation = BM_All_Points_DSM_GE_All[i][j].z;
                tmp_pt.time_phase = i;
                tmp_pt.index = j;

                cloud.pts.push_back(tmp_pt);
            }
        }

        // 构建KD树
        MyKDTree index(2 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        index.buildIndex();

        // 执行半径搜索
        double search_radius = 0.7 * 0.7;  // 0.49
        // Prepare result set
        nanoflann::SearchParameters params;

        //FILE* file = fopen("D:\\VS_Projects\\aux_data\\results.txt", "w");  // 打开文件用于写入
        
        size_t remove_n = 0;

        // 获取平移参数
        double x_GE = 0;
        double y_GE = 0;
        double z_GE = 0;
        size_t GE_model_n = 0;
        
        // 为避免n存在重复使用同一个点的问题，增加判断类型到结构体中
        size_t repeated_n = 0;
        size_t non_repeated_n_1 = 0;
        size_t non_repeated_n_2 = 0;


        // 遍历点并进行搜索
        for (int i = 0; i < BM_All_Points_DSM_GE_All.size(); i++)
        {
            for (int j = 0; j < BM_All_Points_DSM_GE_All[i].size(); j++)
            {
                const auto& query_point = BM_All_Points_DSM_GE_All[i][j];
                double query_pt[2] = { query_point.x, query_point.y };

                std::vector<ResultItem<uint32_t, double>> resultSet;
                size_t nMatches = index.radiusSearch(&query_pt[0], search_radius, resultSet, params);

                repeated_n += nMatches;

                if (nMatches >= 1)
                {
                    // 计算DSM上匹配点平均值和标准差，使用resultSet遍历
                    double GE_mean_x = 0;
                    double GE_mean_y = 0;
                    size_t n = 0;
                    for (int k = 0; k < resultSet.size(); k++)
                    {
                        if (cloud.pts[resultSet[k].first].is_chosen_1 == false)
                        {
                            non_repeated_n_1 += 1;
                            cloud.pts[resultSet[k].first].is_chosen_1 = true;

                            auto& DSM_matched_point = cloud.pts[resultSet[k].first];

                            double delta_x_GE = BM_All_Points_GE_All[DSM_matched_point.time_phase][DSM_matched_point.index].x - GE_mean_x;
                            double delta_y_GE = BM_All_Points_GE_All[DSM_matched_point.time_phase][DSM_matched_point.index].y - GE_mean_y;
                            n++;

                            GE_mean_x += delta_x_GE / n;
                            GE_mean_y += delta_y_GE / n;
                        }
                    }

                    double GE_mean_x_new = 0;
                    double GE_mean_y_new = 0;
                    double GE_mean_z_new = 0;

                    double DSM_mean_x = 0;
                    double DSM_mean_y = 0;
                    double DSM_mean_z = 0;

                    n = 0;
                    for (int k = 0; k < resultSet.size(); k++)
                    {
                        if (cloud.pts[resultSet[k].first].is_chosen_2 == false)
                        {
                            non_repeated_n_2 += 1;
                            cloud.pts[resultSet[k].first].is_chosen_2 = true;

                            auto& DSM_matched_point = cloud.pts[resultSet[k].first];
                            double dist_x = BM_All_Points_GE_All[DSM_matched_point.time_phase][DSM_matched_point.index].x - GE_mean_x;
                            double dist_y = BM_All_Points_GE_All[DSM_matched_point.time_phase][DSM_matched_point.index].y - GE_mean_y;

                            double sigma = sqrt(dist_x * dist_x + dist_y * dist_y);

                            if (sigma > 5)
                            {
                                remove_n++;
                                continue;
                            }        
                            double delta_x_GE = BM_All_Points_GE_All[DSM_matched_point.time_phase][DSM_matched_point.index].x - DSM_matched_point.x - x_GE;
                            double delta_y_GE = BM_All_Points_GE_All[DSM_matched_point.time_phase][DSM_matched_point.index].y - DSM_matched_point.y - y_GE;
                            double delta_z_GE = BM_All_Points_GE_All[DSM_matched_point.time_phase][DSM_matched_point.index].z - DSM_matched_point.elevation - z_GE;

                            GE_model_n++;

                            x_GE += delta_x_GE / GE_model_n;
                            y_GE += delta_y_GE / GE_model_n;
                            z_GE += delta_z_GE / GE_model_n;

                        }      
                    }
                }
            }
        }
        printf("Remove n:%zu\n", remove_n);
        printf("Repeated n: %zu\n", repeated_n);

        printf("Non repeated n: %zu\n", non_repeated_n_1);
        printf("Non repeated n: %zu\n", non_repeated_n_2);

        printf("GE_Model_n: %zu\n", GE_model_n);

        printf("已完成GE的2024方法校正\n");


        printf("Google_Earth_2024:\nArea_Folder     x      y     z\n");
        printf("%s %lf   %lf   %lf\n", Area_Folder[0].c_str(), x_GE, y_GE, z_GE);

        Point3d Model_Params;
        Model_Params.x = x_GE;
        Model_Params.y = y_GE;
        Model_Params.z = z_GE;

        calculateGeoPositioningAccuracy("2024", Model_Params, All_Area_Dir + "\\Truth_GCP.txt", All_Area_Dir + "\\Unrectified_GCP.txt");

        string txt_path = All_Area_Dir + "\\Model_Translate_Parameters_2024.txt";

        FILE* p_Model;
        p_Model = fopen(txt_path.c_str(), "w");

        fprintf(p_Model, "Google_Earth 2024:\nArea_Folder     x      y     z\n");
        fprintf(p_Model, "%s %lf   %lf   %lf\n", All_Area_Dir.c_str(), x_GE, y_GE, z_GE);

        fclose(p_Model);

        //第五步，根据模型参数校正DOM和DSM
        if (0 != _access((All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "Rectified_DOM_DSM_2024").c_str(), 0))
        {
            _mkdir((All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "Rectified_DOM_DSM_2024").c_str());   // 返回 0 表示创建成功，-1 表示失败
        }
        string dom_path_tif = All_Area_Dir + "\\" + Area_Folder[0] + "\\DOM\\orthophoto.tif";
        string dsm_path_tif = All_Area_Dir + "\\" + Area_Folder[0] + "\\DSM\\dsm_i.tif";

        string dom_rect_path_tif_GE_iter = All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "Rectified_DOM_DSM_2024\\orthophoto_Rect_GE.tif";
        DOM_Rectified(dom_path_tif.c_str(), dom_rect_path_tif_GE_iter.c_str(), "GTiff", x_GE, y_GE);
        string dsm_rect_path_tif_GE_iter = All_Area_Dir + "\\" + Area_Folder[0] + "\\Rectified_DOM_DSM_2024\\DSM_Rect_GE.tif";
        //实现DSM的z方向校正
        DSM_Rectified(dsm_path_tif.c_str(), dsm_rect_path_tif_GE_iter.c_str(), "GTiff", x_GE, y_GE, z_GE);
        printf("已完成DSM和DOM的几何校正\n");
    }


    //精度评价
    if (is_Precision_Evaluation == true)
    {
        printf("正在对几何校正后的DSM进行精度评价......\n");

        string TruthFile = All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "TRUTH";
        vector <string> Truth_tif;
        GetAllFiles_tif(TruthFile, Truth_tif);

        string RectFile = All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "Rectified_DOM_DSM_2024";
        vector <string> Rect_tif;
        GetAllFiles_tif(RectFile, Rect_tif);

        string UnRectFile = All_Area_Dir + "\\" + Area_Folder[0] + "\\" + "DSM";
        vector <string> UnRect_tif;
        GetAllFiles_tif(UnRectFile, UnRect_tif);

        //去除校正后DOM的影响，因为精度评价仅需要DSM
        for (int j = 0; j < Rect_tif.size(); j++)
        {
            if (Rect_tif[j].find("DSM") == string::npos)
            {
                Rect_tif.erase(Rect_tif.begin() + j);
                j--;
            }
        }

        //创建文件夹存放精度评价矩阵
        if (0 != _access((All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation").c_str(), 0))
        {
            _mkdir((All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation").c_str());   // 返回 0 表示创建成功，-1 表示失败
        }
        if (0 != _access((All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation_UnRect").c_str(), 0))
        {
            _mkdir((All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation_UnRect").c_str());   // 返回 0 表示创建成功，-1 表示失败
        }

        //对校正后的DSM进行精度评价
        for (int j = 0; j < Rect_tif.size(); j++)
        {
            Precision_Evaluation((TruthFile + "\\" + Truth_tif[0]).c_str(), (RectFile + "\\" + Rect_tif[j]).c_str(), All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation", Area_Folder[0], true);
        }
        //对未校正的DSM进行精度评价
        Precision_Evaluation((TruthFile + "\\" + Truth_tif[0]).c_str(), (UnRectFile + "\\" + UnRect_tif[0]).c_str(), All_Area_Dir + "\\" + Area_Folder[0] + "_Precision_Evaluation_UnRect", Area_Folder[0], true);

        printf("已完成对几何校正后DSM的精度评价\n");
    }
}


// 保存点数据到二进制文件
void savePointsBinary(const std::string& filename, const std::vector<std::vector<Point3d>>& points) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    for (const auto& pointVec : points) {
        size_t size = pointVec.size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        out.write(reinterpret_cast<const char*>(pointVec.data()), size * sizeof(Point3d));
    }
    out.close();
}

// 从二进制文件加载点数据
void loadPointsBinary(const std::string& filename, std::vector<std::vector<Point3d>>& points) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open " << filename << " for reading.\n";
        return;
    }

    points.clear();
    while (true) {
        size_t size = 0;
        // 首先读取向量的大小
        if (!in.read(reinterpret_cast<char*>(&size), sizeof(size))) {
            break;  // 如果读取失败或到达文件末尾，退出循环
        }
        if (size == 0) {
            points.push_back(std::vector<Point3d>());  // 如果大小为0，仍然添加一个空的向量
            continue;  // 继续读取下一个数据块
        }
        // 根据读取的大小创建向量并读取数据
        std::vector<Point3d> pointVec(size);
        if (!in.read(reinterpret_cast<char*>(pointVec.data()), size * sizeof(Point3d))) {
            break;  // 如果数据块的读取失败，退出循环
        }
        points.push_back(pointVec);  // 将读取的向量添加到结果列表中
    }
    in.close();  // 关闭文件
}

bool hasNaN(const Point3d& point) {
    return std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z);
}

void removeNaNPoints(std::vector<std::vector<Point3d>>& points1, std::vector<std::vector<Point3d>>& points2) {
    for (int i = 0; i < points1.size(); i++)
    {
        for (int j = 0; j < points1[i].size(); j++)
        {
            if (hasNaN(points1[i][j]))
            {
                points1[i].erase(points1[i].begin() + j);
                points2[i].erase(points2[i].begin() + j);
                j--;
            }
        }
    }

    for (int i = 0; i < points2.size(); i++)
    {
        for (int j = 0; j < points2[i].size(); j++)
        {
            if (hasNaN(points2[i][j]))
            {
                points1[i].erase(points1[i].begin() + j);
                points2[i].erase(points2[i].begin() + j);
                j--;
            }
        }
    }
}

double calculateRMSE(const std::vector<Point3d>& truth, const std::vector<Point3d>& data) {
    double sum_sq = 0;
    size_t count = truth.size();
    for (size_t i = 0; i < count; ++i) {
        sum_sq += sqrt((truth[i].x - data[i].x) * (truth[i].x - data[i].x) +
            (truth[i].y - data[i].y) * (truth[i].y - data[i].y) +
            (truth[i].z - data[i].z) * (truth[i].z - data[i].z));
    }
    return sum_sq / count;
}

double calculateAverageEuclideanDistance_XY(const std::vector<Point3d>& truth, const std::vector<Point3d>& data) {
    double sum_distances = 0;
    size_t count = truth.size();
    for (size_t i = 0; i < count; ++i) {
        double distance = sqrt((truth[i].x - data[i].x) * (truth[i].x - data[i].x) +
            (truth[i].y - data[i].y) * (truth[i].y - data[i].y));
        sum_distances += distance;
    }
    return sum_distances / count; // 返回平均距离
}

double calculateRMSE_Z(const std::vector<Point3d>& truth, const std::vector<Point3d>& data) {
    double sum_sq = 0;
    size_t count = truth.size();
    for (size_t i = 0; i < count; ++i) {
        sum_sq += sqrt((truth[i].z - data[i].z) * (truth[i].z - data[i].z));
    }
    return sum_sq / count;
}

void loadPoints(const std::string& filename, std::vector<Point3d>& points) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    std::getline(file, line); // Read header line
    Point3d point;
    while (file >> point.x >> point.y >> point.z) {
        points.push_back(point);
    }
    file.close();
}

void applyCorrection(std::vector<Point3d>& data, const Point3d& correction) {
    for (auto& point : data) {
        point.x += correction.x;
        point.y += correction.y;
        point.z += correction.z;
    }
}

void calculateGeoPositioningAccuracy(const std::string& method, const Point3d& correction,
    const std::string& truthPath, const std::string& dataPath) {
    std::vector<Point3d> truth, data;
    loadPoints(truthPath, truth);
    loadPoints(dataPath, data);

    if (truth.size() != data.size()) {
        throw std::runtime_error("The number of points in truth and data files do not match.");
    }
    double rmse_before_xyz = calculateRMSE(truth, data);
    double rmse_before_xy = calculateAverageEuclideanDistance_XY(truth, data);
    double rmse_before_z = calculateRMSE_Z(truth, data);

    applyCorrection(data, correction);

    double rmse_after_xyz = calculateRMSE(truth, data);
    double rmse_after_xy = calculateAverageEuclideanDistance_XY(truth, data);
    double rmse_after_z = calculateRMSE_Z(truth, data);

    std::cout << "Method: " << method << std::endl;
    std::cout << "RMSE before correction (XY): " << rmse_before_xy << std::endl;
    std::cout << "RMSE before correction (Z): " << rmse_before_z << std::endl;
    std::cout << "RMSE before correction (XYZ): " << rmse_before_xyz << std::endl;
    std::cout << "RMSE after correction (XY): " << rmse_after_xy << std::endl;
    std::cout << "RMSE after correction (Z): " << rmse_after_z << std::endl;
    std::cout << "RMSE after correction (XYZ): " << rmse_after_xyz << std::endl;
}

void RANSAC_Point_Semantic_Constraint(vector <vector<Point3d>>& BM_pt, vector <vector<Point3d>>& DSM_pt, vector <vector<float>> Building_Prob_Weight, int Init_Iter_Time, double Threshold, double confidence, double Inlier_Ratio, Point3d& Model_Translation_Parameter_RANSAC, string BM_name, string dirpath,
    string Mask_Path, string Mask_Type, string BM_Type, int Cut_Num)
{
    size_t total_pt_num = 0;
    for (int i = 0; i < BM_pt.size(); i++)
    {
        total_pt_num += BM_pt.size();
    }

    if (Init_Iter_Time > total_pt_num)
    {
        Init_Iter_Time = total_pt_num;
    }

    Building_Probabilty_Weighting(BM_pt, DSM_pt, Building_Prob_Weight, Mask_Path, Mask_Type, BM_Type, Cut_Num);

    int all_bef_RANSAC = 0;
    for (int ri = 0; ri < BM_pt.size(); ri++)
    {
        all_bef_RANSAC += BM_pt[ri].size();
    }
    printf("RANSAC剔除前点数 %d", all_bef_RANSAC);

    // 创建RANSAC前后点文件夹
    if (0 != _access((dirpath + "\\" + "RANSAC_pt_res").c_str(), 0))
    {
        int result = _mkdir((dirpath + "\\" + "RANSAC_pt_res").c_str());   // 返回 0 表示创建成功，-1 表示失败
        if (result != 0) {
            switch (errno) {
            case ENOENT:
                std::cerr << "路径错误！\n" << std::endl;
                break;
            default:
                std::cerr << "创建文件夹时遇到未知错误，请检查文件夹权限！\n" << std::endl;
                break;
            }
        }
    }

    // 输出一个RANSAC前的所有DSM_pt点坐标的txt文件，此处为地理位置的x，y
    FILE* p_bef;
    string RANSAC_txt_bef = dirpath + "\\RANSAC_pt_res\\RANSAC_bef_" + BM_name + ".txt";
    p_bef = fopen(RANSAC_txt_bef.c_str(), "w");

    for (int ii = 0; ii < BM_pt.size(); ii++)
    {
        for (int jj = 0; jj < BM_pt[ii].size(); jj++)
        {
            if (BM_pt[ii].size() == 0)
            {
                continue;
            }
            if (isnan(DSM_pt[ii][jj].z) || isnan(BM_pt[ii][jj].z))
            {
                BM_pt[ii].erase(BM_pt[ii].begin() + jj);
                DSM_pt[ii].erase(DSM_pt[ii].begin() + jj);
                jj--;
                continue;
            }
            fprintf(p_bef, "%lf %lf\n", DSM_pt[ii][jj].x, DSM_pt[ii][jj].y);
        }
    }

    fclose(p_bef);

    //所有匹配对的整体RANSAC模型
    int iter_time = Init_Iter_Time;
    int Max_Count = 0;
    for (int i = 0; i < BM_pt.size(); i++)
    {
        Max_Count += BM_pt[i].size();
    }
    int Good_Model_key1 = 0;
    int Good_Model_key2 = 0;
    int Good_Model_Count = 0;
    double Model_Para[3];//依次为x，y，z的平移量，为BM_pt - DSM_pt(参考真值减观测值)

    //随机取点，计算平移模型参数
    vector<int> rand_num1;//注意erase的时候不要把整个匹配对都删除了
    vector<int> rand_num2;
    weightedRandomSample(Building_Prob_Weight, rand_num1, rand_num2, iter_time);

    for (int i = 0; i < iter_time; i++)
    {
        Model_Para[0] = BM_pt[rand_num1[i]][rand_num2[i]].x - DSM_pt[rand_num1[i]][rand_num2[i]].x;
        Model_Para[1] = BM_pt[rand_num1[i]][rand_num2[i]].y - DSM_pt[rand_num1[i]][rand_num2[i]].y;
        Model_Para[2] = BM_pt[rand_num1[i]][rand_num2[i]].z - DSM_pt[rand_num1[i]][rand_num2[i]].z;

        //计算内点数量
        int inlier_count = 0;
        for (int ii = 0; ii < BM_pt.size(); ii++)
        {
            for (int jj = 0; jj < BM_pt[ii].size(); jj++)
            {
                double temp_x = DSM_pt[ii][jj].x + Model_Para[0];
                double temp_y = DSM_pt[ii][jj].y + Model_Para[1];
                double temp_z = DSM_pt[ii][jj].z + Model_Para[2];
                double dx = abs(temp_x - BM_pt[ii][jj].x);
                double dy = abs(temp_y - BM_pt[ii][jj].y);
                double dz = abs(temp_z - BM_pt[ii][jj].z);

                double dist = sqrt(dx * dx + dy * dy + dz * dz);

                if (dist < Threshold) inlier_count++;
            }
        }

        if (inlier_count > round(Max_Count * Inlier_Ratio))//更新迭代次数
        {
            double p = double(inlier_count) / Max_Count;
            double numerator = log(1 - confidence);
            double deminator = log(1 - p);//模型仅需一个点，故为p的一次方
            double temp_iter_time = ceil(numerator / deminator);
            if (temp_iter_time <= i)
            {
                iter_time = int(ceil(i + temp_iter_time));
            }
            else iter_time = int(ceil(temp_iter_time));
        }
        //2023.1.13修改，为避免没有满足比例阈值直接选第一个点对的情况
        if (inlier_count > Good_Model_Count)
        {
            Good_Model_Count = inlier_count;
            Good_Model_key1 = rand_num1[i];
            Good_Model_key2 = rand_num2[i];
        }
    }
    Model_Para[0] = BM_pt[Good_Model_key1][Good_Model_key2].x - DSM_pt[Good_Model_key1][Good_Model_key2].x;
    Model_Para[1] = BM_pt[Good_Model_key1][Good_Model_key2].y - DSM_pt[Good_Model_key1][Good_Model_key2].y;
    Model_Para[2] = BM_pt[Good_Model_key1][Good_Model_key2].z - DSM_pt[Good_Model_key1][Good_Model_key2].z;
    //删除外点，两个点集都删除
    for (int ii = 0; ii < BM_pt.size(); ii++)
    {
        for (int jj = 0; jj < BM_pt[ii].size(); jj++)
        {
            double temp_x = DSM_pt[ii][jj].x + Model_Para[0];
            double temp_y = DSM_pt[ii][jj].y + Model_Para[1];
            double temp_z = DSM_pt[ii][jj].z + Model_Para[2];
            double dx = abs(temp_x - BM_pt[ii][jj].x);
            double dy = abs(temp_y - BM_pt[ii][jj].y);
            double dz = abs(temp_z - BM_pt[ii][jj].z);

            double dist = sqrt(dx * dx + dy * dy + dz * dz);
            //删除外点
            if (dist > Threshold)
            {
                BM_pt[ii].erase(BM_pt[ii].begin() + jj);
                DSM_pt[ii].erase(DSM_pt[ii].begin() + jj);
                jj--;
            }
        }
    }

    int total_inlier_num = 0;
    for (int i = 0; i < BM_pt.size(); i++)
    {
        total_inlier_num += BM_pt[i].size();
    }
    printf(" RANSAC剔除后点数 %d ", total_inlier_num);
    //printf("内外点比例为：%lf\n", float(total_inlier_num) / all_bef_RANSAC);

    Model_Para[0] = 0;
    Model_Para[1] = 0;
    Model_Para[2] = 0;


    for (int ii = 0; ii < BM_pt.size(); ii++)
    {
        for (int jj = 0; jj < BM_pt[ii].size(); jj++)
        {
            Model_Para[0] += BM_pt[ii][jj].x - DSM_pt[ii][jj].x;
            Model_Para[1] += BM_pt[ii][jj].y - DSM_pt[ii][jj].y;
            Model_Para[2] += BM_pt[ii][jj].z - DSM_pt[ii][jj].z;
        }
    }

    // 输出一个RANSAC后的所有DSM_pt点坐标的txt文件，此处为地理位置的x，y
    FILE* p_aft;
    string RANSAC_txt_aft = dirpath + "\\RANSAC_pt_res\\RANSAC_aft_" + BM_name + ".txt";
    p_aft = fopen(RANSAC_txt_aft.c_str(), "w");

    for (int ii = 0; ii < BM_pt.size(); ii++)
    {
        for (int jj = 0; jj < BM_pt[ii].size(); jj++)
        {
            fprintf(p_aft, "%lf %lf\n", DSM_pt[ii][jj].x, DSM_pt[ii][jj].y);
        }
    }
    fclose(p_aft);

    Model_Para[0] /= total_inlier_num;
    Model_Para[1] /= total_inlier_num;
    Model_Para[2] /= total_inlier_num;
    Point3d Model_Para_tmp;
    Model_Para_tmp.x = Model_Para[0];
    Model_Para_tmp.y = Model_Para[1];
    Model_Para_tmp.z = Model_Para[2];
    Model_Translation_Parameter_RANSAC = Model_Para_tmp;

}

void Building_Probabilty_Weighting(vector <vector<Point3d>>& BM_pt, vector <vector<Point3d>>& DSM_pt, vector <vector<float>>& Building_Prob_Weight, string Mask_Path, string Mask_Type, string BM_Type, int Cut_Num)
{
    if (Mask_Type == "DOM")
    {
        vector <string> files;
        GetAllFiles_tif(Mask_Path, files);

        if (files.size() != 1)
        {
            printf("若为DOM建筑物掩膜，只应有一个掩膜文件，请检查输入文件夹和掩膜类型！\n");
            return;
        }
        string filepath = Mask_Path + "\\" + files[0];

        GDALDataset* poDataset;

        GDALAllRegister(); // 注册所有的驱动程序

        // 打开文件
        poDataset = (GDALDataset*)GDALOpen(filepath.c_str(), GA_ReadOnly);
        if (poDataset == nullptr) {
            printf("无法打开掩膜文件！\n");
            return;
        }
         
        // 获取坐标变换系数
        double trans[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
        if (poDataset->GetGeoTransform(trans) != CE_None) //影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。
        {
            printf("无法获取掩膜文件的地理信息！\n");
            GDALClose(poDataset);
            return;
        }
        trans[0] += 1.0 / 2 * trans[1];
        trans[3] += 1.0 / 2 * trans[5];

        GDALRasterBand* poBand = poDataset->GetRasterBand(1);

        for (int ii = 0; ii < BM_pt.size(); ii++)
        {
            for (int jj = 0; jj < BM_pt[ii].size(); jj++)
            {
                // 计算行列号
                double dTemp = trans[2] * trans[4] - trans[1] * trans[5];
                double dRow = (trans[4] * (DSM_pt[ii][jj].x - trans[0]) - trans[1] * (DSM_pt[ii][jj].y - trans[3])) / dTemp;
                double dCol = (trans[2] * (DSM_pt[ii][jj].y - trans[3]) - trans[5] * (DSM_pt[ii][jj].x - trans[0])) / dTemp;
                int dx = int(dCol + 0.5);
                int dy = int(dRow + 0.5);
                float value;
                poBand->RasterIO(GF_Read, dx, dy, 1, 1, &value, 1, 1, GDT_Float32, 0, 0);

                Building_Prob_Weight[ii][jj] = 1.0 - value;

            }
            if (BM_pt[ii].size() == 0)
            {
                BM_pt.erase(BM_pt.begin() + ii);
                DSM_pt.erase(DSM_pt.begin() + ii);
                ii--;
            }
        }

        // 关闭数据集
        GDALClose(poDataset);
    }

    if (Mask_Type == "BASEMAP")
    {
        vector <string> files;
        string Mask_Path_BM = Mask_Path + "\\" + BM_Type;
        GetAllFiles_tif(Mask_Path_BM, files);

        if (files.size() == 0)
        {
            printf("无掩膜文件，请检查输入文件夹！\n");
            return;
        }

        int BM_index = 0;
        int BM_max_num = files.size();
        for (int ii = 0; ii < BM_pt.size(); ii++)
        {
            if (BM_pt[ii].size() == 0)
            {
                continue;
            }

            string filepath = Mask_Path_BM + "\\" + files[BM_index];

            GDALDataset* poDataset;

            GDALAllRegister(); // 注册所有的驱动程序

            // 打开文件
            poDataset = (GDALDataset*)GDALOpen(filepath.c_str(), GA_ReadOnly);
            if (poDataset == nullptr) {
                printf("无法打开掩膜文件！");
                printf("%s\n", files[BM_index].c_str());
                return;
            }

            // 获取坐标变换系数
            double trans[6] = { 0,1,0,0,0,1 };//定义为默认值，即x、y分辨率为1，其他信息为0 
            if (poDataset->GetGeoTransform(trans) != CE_None) //影像左上角横坐标：geoTransform[0]，影像左上角纵坐标：geoTransform[3]，遥感图像的水平空间分辨率为geoTransform[1]，遥感图像的垂直空间分辨率为geoTransform[5]，如果遥感影像方向没有发生旋转，geoTransform[2] 与 row* geoTransform[4] 为零。
            {
                printf("无法获取掩膜文件的地理信息！\n");
                GDALClose(poDataset);
                return;
            }

            GDALRasterBand* poBand = poDataset->GetRasterBand(1);
            // 我现在cutnum为4，有9张底图，那么我的BM_pt的数量就是4*9=36，我需要判断：在什么区间内我使用哪一张底图,BM的index是0到8，0、1、2、3..7的时候是bm0，8-15的时候是bm1


            for (int jj = 0; jj < BM_pt[ii].size(); jj++)
            {
                // 计算行列号
                double dTemp = trans[2] * trans[4] - trans[1] * trans[5];
                double dRow = (trans[4] * (BM_pt[ii][jj].x - trans[0]) - trans[1] * (BM_pt[ii][jj].y - trans[3])) / dTemp;
                double dCol = (trans[2] * (BM_pt[ii][jj].y - trans[3]) - trans[5] * (BM_pt[ii][jj].x - trans[0])) / dTemp;
                int dx = int(dCol + 0.5);
                int dy = int(dRow + 0.5);
                float value;
                poBand->RasterIO(GF_Read, dx, dy, 1, 1, &value, 1, 1, GDT_Float32, 0, 0);

                if (isnan(value))
                {
                    Building_Prob_Weight[ii][jj] = 0.0;
                }
                else 
                {
                    // 赋予权重
                    Building_Prob_Weight[ii][jj] = 1.0 - value;
                }
            }

            // 关闭数据集
            GDALClose(poDataset);
            if (BM_index < BM_max_num - 1)
            {
                BM_index++;
            }
            else
            {
                BM_index = 0;
            }
        }
    }
}

//void weightedRandomSample(const std::vector<std::vector<float>>& weights, vector<int>& random_1, vector<int>& random_2, int group_num)
//{
//    for (int rand = 0; rand < group_num; rand++)
//    {
//        // 将二维权重矩阵转换为一维，同时构建累积权重数组
//        std::vector<float> cumulativeWeights;
//        float totalWeight = 0.0f;
//
//        for (size_t i = 0; i < weights.size(); ++i) {
//            for (size_t j = 0; j < weights[i].size(); ++j) {
//                totalWeight += weights[i][j];
//                cumulativeWeights.push_back(totalWeight);
//            }
//        }
//
//        // 生成随机数
//        std::mt19937 gen(rand);
//        std::uniform_real_distribution<float> dis(0.0f, totalWeight);
//        float rnd = dis(gen);
//
//        // 使用二分查找找到对应的索引
//        auto lower_bound_it = std::lower_bound(cumulativeWeights.begin(), cumulativeWeights.end(), rnd);
//        size_t index = std::distance(cumulativeWeights.begin(), lower_bound_it);
//
//        // 计算二维索引
//        int tmp_random_1 = 0;
//        int tmp_random_2 = 0;
//        size_t acc = weights[0].size();
//        while (index >= acc && tmp_random_1 + 1 < weights.size()) {
//            tmp_random_1++;
//            acc += weights[tmp_random_1].size();
//        }
//        tmp_random_2 = index - (acc - weights[tmp_random_1].size());
//        random_1.push_back(tmp_random_1);
//        random_2.push_back(tmp_random_2);
//    }
//}

void weightedRandomSample(const std::vector<std::vector<float>>& weights, std::vector<int>& random_1, std::vector<int>& random_2, int group_num) {
    // 使用固定的种子初始化随机数生成器
    std::mt19937 gen(1024);  // 例如使用种子42

    size_t total_pt_num = 0;
    for (int i = 0; i < weights.size(); i++)
    {
        total_pt_num += weights.size();
    }

    if (group_num > total_pt_num)
    {
        group_num = total_pt_num;
    }

    for (int rand = 0; rand < group_num; rand++) {
        std::vector<float> cumulativeWeights;
        float totalWeight = 0.0f;

        // 构建累积权重数组
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                totalWeight += weights[i][j];
                cumulativeWeights.push_back(totalWeight);
            }
        }

        bool isUnique;
        int tmp_random_1, tmp_random_2;

        do {
            // 生成随机数
            std::uniform_real_distribution<float> dis(0.0f, totalWeight);
            float rnd = dis(gen);

            // 使用二分查找找到对应的索引
            auto lower_bound_it = std::lower_bound(cumulativeWeights.begin(), cumulativeWeights.end(), rnd);
            size_t index = std::distance(cumulativeWeights.begin(), lower_bound_it);

            // 计算二维索引
            tmp_random_1 = 0;
            size_t acc = weights[0].size();
            while (index >= acc && tmp_random_1 + 1 < weights.size()) {
                tmp_random_1++;
                acc += weights[tmp_random_1].size();
            }
            tmp_random_2 = index - (acc - weights[tmp_random_1].size());

            // 检查是否唯一
            isUnique = true;
            for (size_t i = 0; i < random_1.size(); i++) {
                if (random_1[i] == tmp_random_1 && random_2[i] == tmp_random_2) {
                    isUnique = false;
                    break;
                }
            }
        } while (!isUnique); // 如果不唯一，重新生成

        // 保存唯一索引
        random_1.push_back(tmp_random_1);
        random_2.push_back(tmp_random_2);
    }
}