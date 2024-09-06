#include <vector>
#include <iostream>
#include <stdio.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
// #include "sophus/se3.h"
#include <sophus/se3.hpp>
using namespace cv;
using namespace std;
// 相机内参
Eigen::Matrix<double ,3,3> camMatrix;
// BA最大迭代次数
const int MAX_LOOP=5;

/***
 *
 * @param P 3Dd点的坐标
 * @param Pose 相机位姿
 * @return 2x6的雅克比矩阵
 */
Eigen::Matrix<double ,2,6> findPoseJacobian(Eigen::Matrix<double,4,4> Pose,Eigen::Vector3d P){
    Eigen::Matrix<double ,2,6> ans;
    Eigen::Matrix<double ,4,1>  Point;
    Point(0,0)=P(0);
    Point(1,0)=P(1);
    Point(2,0)=P(2);
    Point(3,0)=1.0;
    Eigen::Matrix<double ,4,1>  cam_Point=Pose*Point; //计算3D点在相机坐标系下的坐标
    double fx=camMatrix(0,0);
    double fy=camMatrix(1,1);
    double x=cam_Point(0,0);
    double y=cam_Point(1,0);
    double z=cam_Point(2,0);
    ans(0,0)=fx/z;
    ans(0,1)=0;
    ans(0,2)=-1*((fx*x)/(z*z));
    ans(0,3)=-1*((fx*x*y)/(z*z));
    ans(0,4)=fx+(fx*x*x)/(z*z);
    ans(0,5)=-1*((fx*y)/z);
    ans(1,0)=0;
    ans(1,1)=fy/z;
    ans(1,2)=-1*((fy*y)/(z*z));
    ans(1,3)=-fy-((fy*y*y)/(z*z));
    ans(1,4)=(fy*x*y)/(z*z);
    ans(1,5)=(fy*x)/z;
    //注意此处的-1,参见《14讲》p164
    return -1*ans;
}
/***
 *
 * @param P 3D点的坐标
 * @param Pose 相机位姿
 * @return  2x3的雅克比矩阵
 */
Eigen::Matrix<double ,2,3> findPointJacobian(Eigen::Matrix<double,4,4> Pose,Eigen::Vector3d P){
    Eigen::Matrix<double ,2,3> ans;
    Eigen::Matrix<double ,4,1> Point;
    Point(0,0)=P(0);
    Point(1,0)=P(1);
    Point(2,0)=P(2);
    Point(3,0)=1.0;
    Eigen::Matrix<double ,4,1>  cam_Point=Pose*Point; //计算3D点在相机坐标系下的坐标
    double fx=camMatrix(0,0);
    double fy=camMatrix(1,1);
    double x=cam_Point(0,0);
    double y=cam_Point(1,0);
    double z=cam_Point(2,0);
    ans(0,0)=fx/z;
    ans(0,1)=0;
    ans(0,2)=-1*((fx*x)/(z*z));
    ans(1,0)=0;
    ans(1,1)=fy/z;
    ans(1,2)=-1*((fy*y)/(z*z));
    return -1*ans*Pose.block<3,3>(0,0);
}
/***
 *
 * @param x 状态量
 * @return 整体的雅克比矩阵
 */
Eigen::MatrixXd findWholeJacobian(Eigen::MatrixXd x)
{
    Eigen::MatrixXd ans;
    int size_P=(int)(x.rows()-6)/3;
    ans.resize(2*size_P,6+3*size_P);
    ans.setZero();
    Eigen::VectorXd v_temp(6);
    v_temp=x.block(0,0,6,1);
    Sophus::SE3<double> SE3_temp=Sophus::SE3<double>::exp(v_temp);
    Eigen::Matrix<double,4,4> Pose = SE3_temp.matrix();
    std::cout<<"size_P="<<size_P<<std::endl;//65
    for(int i=0;i<size_P;i++){
        //Block of size (p,q), starting at (i,j)
        //matrix.block(i,j,p,q)->针对动态矩阵,matrix.block<p,q>(i,j)->针对静态矩阵
        ans.block(2*i,0,2,6)=findPoseJacobian(Pose,x.block(6+3*i,0,3,1));
        ans.block(i*2,6+3*i,2,3)=findPointJacobian(Pose,x.block(6+3*i,0,3,1));
    }

    std::cout<<"--DEBUG--"<<"findWholeJacobian end"<<std::endl;
    std::cout<<"J.cols() = "  <<ans.cols()<<endl;
    std::cout<<"J.rows() = "<<ans.rows()<<endl;
    return ans;
}
/***
 *
 * @param x 状态量
 * @param v_Point2D 观测到特征点的像素坐标
 * @return f(x)
 */
Eigen::Matrix<double ,Eigen::Dynamic,1> findCostFunction(Eigen::MatrixXd x, std::vector<cv::Point2d> v_Point2D)
{
    //e=u-K*T*P; u为图像上的观测坐标,K为相机内参,T为相机外参，P为3D点坐标;
    double fx=camMatrix(0,0);
    double fy=camMatrix(1,1);
    double cx=camMatrix(0,2);
    double cy=camMatrix(1,2);
    Eigen::Matrix<double ,Eigen::Dynamic,1> ans;

    int size_P=(int)(x.rows()-6)/3;

    if(size_P!=v_Point2D.size()){
        std::cout<<"---ERROR---"<<endl;
        return ans;
    }
    //把李代数转化为矩阵 Pose为变换矩阵
    Eigen::VectorXd v_temp(6);
    v_temp=x.block(0,0,6,1);
    Sophus::SE3<double> SE3_temp=Sophus::SE3<double>::exp(v_temp);
    Eigen::Matrix<double,4,4> Pose = SE3_temp.matrix();

    ans.resize(2*size_P,1);
    ans.setZero();
    for(int i=0;i<size_P;i++){
        Eigen::Matrix<double ,4,1> Point;
        Point(0,0)=x(6+i*3  ,0);
        Point(1,0)=x(1+6+i*3,0);
        Point(2,0)=x(2+6+i*3,0);
        Point(3,0)=1.0;
        //计算e
        Eigen::Matrix<double ,4,1>  cam_Point=Pose*Point; //计算3D点在相机坐标系下的坐标
        double cam_x=cam_Point(0,0); //相机坐标喜下3D点的坐标
        double cam_y=cam_Point(1,0);
        double cam_z=cam_Point(2,0);
        ans(2*i,  0) = v_Point2D[i].x-((fx*cam_x)/cam_z)-cx ;
        ans(2*i+1,0) = v_Point2D[i].y-((fy*cam_y)/cam_z)-cy ;
    }
    return ans;
}
/**
 *
 * @param v_P3d 一组匹配成功的3D点坐标
 * @param v_P2d 3D坐标点在某帧下的坐标
 * @param t     迭代的初值(4x4矩阵)
 */
void bundleAdjustment(std::vector<cv::Point3d> v_P3d,std::vector<cv::Point2d> v_P2d,Eigen::Matrix<double,4,4> T,cv::Mat img){
    std::cout<<"Do BA by yourself, v1.0"<<std::endl;
    Eigen::Matrix<double ,Eigen::Dynamic,Eigen::Dynamic> x; //状态量 x
    x.resize(6+3*v_P3d.size(),1);  //位姿的李代数+3D点的坐标
    x.setZero();
    //状态量x初始化
    Eigen::Matrix<double ,3,3> R=T.block<3,3>(0,0);
    Eigen::Matrix<double ,3,1> t=T.block<3,1>(0,3);
    Sophus::SE3<double> XI(R,t);//位姿->ξ
    //cout<<"位姿："<<XI.log().transpose()<<endl;
    x.block(0,0,6,1)=XI.log();       //位姿的李代数
    for(int i=0;i<v_P3d.size();i++){ //3D点坐标

        x(6+3*i,  0)=v_P3d[i].x;
        x(6+3*i+1,0)=v_P3d[i].y;
        x(6+3*i+2,0)=v_P3d[i].z;
        //cout<<"x:"<<v_P3d[i].x<<",y:"<<v_P3d[i].y<<",z:"<<v_P3d[i].z<<endl;
    }

    for(int i=1;i<=MAX_LOOP;i++){ //循环求解BA
        std::cout<<"\033[32m"<<"Doing BA Please wait......"<<std::endl;
        double t = (double)cv::getTickCount(); //计时开始
        Eigen::MatrixXd Jacobian=findWholeJacobian(x);       //求解状态x的Jacobian
        Eigen::MatrixXd JacobianT=Jacobian.transpose();      //求解Jacobian 的转置
        Eigen::MatrixXd H=JacobianT*Jacobian;                //求解H矩阵
        //std::cout<<"H = "<<endl<<H<<endl;
        Eigen::MatrixXd fx=findCostFunction(x,v_P2d);        //求解f(x)在状态x下的值
        Eigen::VectorXd g=-1*JacobianT*fx;                   //求解g,相见<十四讲>p247
        //求解delt_x
        //1.Using the SVD decomposition
        //Eigen::MatrixXd delt_x=H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(g);
        //2.Using the QR decomposition
        std::cout<<"Solving ......"<<"\033[37m"<<std::endl;
        Eigen::MatrixXd delt_x=H.colPivHouseholderQr().solve(g);
        ///李代数相加需要添加一些余项，转化为R再相乘，代替加法,详见14讲 72页；
        ///把SE3上的李代数转化为4x4矩阵
        Eigen::Matrix4d Pos_Matrix = Sophus::SE3<double>::exp(x.block(0,0,6,1)).matrix();
        Eigen::Matrix4d Pos_update_Matrix = Sophus::SE3<double>::exp(delt_x.block(0,0,6,1)).matrix();
        ///矩阵更新
        Pos_Matrix = Pos_Matrix * Pos_update_Matrix;
        ///转化为李代数
        Sophus::SE3<double> new_Pos_se = Sophus::SE3<double>(Pos_Matrix.block<3,3>(0,0),Pos_Matrix.block<3,1>(0,3));
        ///更新姿态
        x = x + delt_x;
        x.block(0,0,6,1)=new_Pos_se.log();
        printf("BA cost %f ms \n", (1000*(cv::getTickCount() - t) / cv::getTickFrequency()));
        //--------------------在原图相上画出观测和预测的坐标-------------------------------------
        Eigen::VectorXd v_temp(6);
        v_temp=x.block(0,0,6,1);
        Sophus::SE3<double> SE3_temp=Sophus::SE3<double>::exp(v_temp);
        Eigen::Matrix<double,4,4> Pose = SE3_temp.matrix();
        cout<<"POSE:"<<endl<<Pose<<endl;
        cv::Mat temp_Mat=img.clone();
        /// 投影到图像上，展现优化效果
        for(int j=0;j<v_P3d.size();j++) {
            double fx=camMatrix(0,0);
            double fy=camMatrix(1,1);
            double cx=camMatrix(0,2);
            double cy=camMatrix(1,2);
            Eigen::Matrix<double, 4, 1> Point;
            //Point(0,0)=v_P3d[j].x;
            //Point(1,0)=v_P3d[j].y;
            //Point(2,0)=v_P3d[j].z;
            Point(0,0)=x(6+3*j,  0);
            Point(1,0)=x(1+6+3*j,  0);
            Point(2,0)=x(2+6+3*j,  0);
            Point(3,0)=1.0;
            Eigen::Matrix<double, 4, 1> cam_Point = Pose * Point; //计算3D点在相机坐标系下的坐标
            cv::Point2d temp_Point2d;
            temp_Point2d.x=(fx*cam_Point(0,0)/cam_Point(2,0))+cx;
            temp_Point2d.y=(fy*cam_Point(1,0)/cam_Point(2,0))+cy;
            cv::circle(temp_Mat,temp_Point2d,3,cv::Scalar(0,0,255),2);
            cv::circle(temp_Mat,v_P2d[j],    2,cv::Scalar(255,0,0),2);
        }
        imshow("REPROJECTION ERROR DISPLAY",temp_Mat);
        cout<<"\033[32m"<<"Iteration： "<<i<<" Finish......"<<"\033[37m"<<endl;
        cout<<"\033[32m"<<"Blue is observation......"<<"\033[37m"<<endl;
        cout<<"\033[32m"<<"Red is reprojection......"<<"\033[37m"<<endl;
        cout<<"\033[32m"<<"Press Any Key to continue......"<<"\033[37m"<<endl;
        cv::waitKey(0);
    }
}
void show(){

}

int main (int argc, char * argv[]) {
    if(argc!=4) {
        cout<<"\033[31m"<<"INPUT ERROR !"<<"\033[32m"<<"Please Input Like: 1.png 2.png 3.png  --(1 is the first image,2 is the second ,3 is depth image)"<<"\033[37m"<<endl;
        return -1;
    }
    string imag1 = argv[1];
    string imag2 = argv[2];
    string imag3 = argv[3];
    //-- 设置相机内参
    camMatrix(0,0)=525.0;
    camMatrix(1,1)=525.0;
    camMatrix(0,2)=319.5;
    camMatrix(1,2)=239.5;
    camMatrix(2,2)=1.0;
    //-- 读取图像
    Mat img_1 = imread (imag1);
    Mat img_2 = imread (imag2);
    Mat img_depth = imread(imag3);
    //-- 初始化
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create(500);
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

    //-- 第一步:检测 Oriented FAST 角点位置
    double t = (double)cv::getTickCount();
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );
    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );
    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> matches;
    matcher->match ( descriptors_1, descriptors_2, matches );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector< DMatch > good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }

    //-- 第五步:绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    printf("ORB detect cost %f ms \n", (1000*(cv::getTickCount() - t) / cv::getTickFrequency()));
    cout<<"good_match = "<<good_matches.size()<<endl;

    std::vector<cv::Point2d> points1,points2;
    std::vector< DMatch > good_matches2;
    std::vector<cv::Point3d> v_3DPoints;
    for ( int i = 0; i < ( int ) good_matches.size(); i++ )
    {
        ///
        if(img_depth.at<ushort>(keypoints_1[good_matches[i].queryIdx].pt)/5000!=0 &&
         !isnan( img_depth.at<ushort>(keypoints_1[good_matches[i].queryIdx].pt)) &&
         !isinf( img_depth.at<ushort>(keypoints_1[good_matches[i].queryIdx].pt)))
        {
            points1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
            points2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
            good_matches2.emplace_back(good_matches[i]);

            //求解3D点坐标，参见TUM数据集
            cv::Point3d temp;
            double  u=keypoints_1[good_matches[i].queryIdx].pt.x;
            double v=keypoints_1[good_matches[i].queryIdx].pt.y;
            temp.z=img_depth.at<ushort>(keypoints_1[good_matches[i].queryIdx].pt)/5000;
            //if(temp.z==0) temp.z = 1;
            temp.x=(u-camMatrix(0,2))*temp.z/camMatrix(0,0);
            temp.y=(v-camMatrix(1,2))*temp.z/camMatrix(1,1);
            v_3DPoints.emplace_back(temp);
        }
    }
    //cout<<"points1.size:"<<points1.size()<<endl;
    //cout<<"points2.size:"<<points2.size()<<endl;
    cout<<"v_3DPoints.size:"<<v_3DPoints.size()<<endl;
    //使用OpenCV提供的代数方法求解：2D-2D
    Point2d principal_point ( 319.5, 239.5);	//相机光心, TUM dataset标定值
    double focal_length = 525;			        //相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
    cv::Mat R,tt;
    recoverPose ( essential_matrix, points1, points2, R, tt, focal_length, principal_point );
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<tt<<endl;
    // -- 开始BA优化求解：3D-2D
    Eigen::Matrix<double,4,4> init;
    init.setIdentity();
    bundleAdjustment(v_3DPoints,points2,init,img_2);

    return 0;
}
