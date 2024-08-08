// Copyright 2024 Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LOAM_FEATURE_LOCALIZATION__LOAM_FEATURE_LOCALIZATION_HPP_
#define LOAM_FEATURE_LOCALIZATION__LOAM_FEATURE_LOCALIZATION_HPP_

#include "utils.hpp"

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <boost/filesystem.hpp>

#include <cv_bridge/cv_bridge.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>

#include <deque>
#include <memory>
#include <string>

namespace loam_feature_localization
{
const int queueLength = 2000;

class LoamFeatureLocalization : public rclcpp::Node
{
public:
  using SharedPtr = std::shared_ptr<LoamFeatureLocalization>;
  using ConstSharedPtr = const std::shared_ptr<LoamFeatureLocalization>;

  explicit LoamFeatureLocalization(const rclcpp::NodeOptions & options);

private:
  // parameters
  std::string point_cloud_topic_;
  std::string imu_topic_;
  std::string odom_topic_;
  std::string corner_map_path_;
  std::string surface_map_path_;
  double lidar_imu_x_;
  double lidar_imu_y_;
  double lidar_imu_z_;
  double lidar_imu_roll_;
  double lidar_imu_pitch_;
  double lidar_imu_yaw_;
  double lidar_min_range_;
  double lidar_max_range_;
  int N_SCAN_;
  int Horizon_SCAN_;
  double edge_threshold_;
  double surface_threshold_;

  double odometry_surface_leaf_size_;
  double mapping_corner_leaf_size_;
  double mapping_surf_leaf_size_;
  double surrounding_key_frame_size_;
  double edge_feature_min_valid_num_;
  double surf_feature_min_valid_num_;
  double imu_rpy_weight_;
  double rotation_tollerance_;
  double z_tollerance_;
  double surrounding_key_frame_adding_angle_threshold_;
  double surrounding_key_frame_adding_dist_threshold_;
  double surrounding_key_frame_density_;
  double surrounding_key_frame_search_radius_;

  Utils::SharedPtr utils;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloud;
  rclcpp::CallbackGroup::SharedPtr callbackGroupLidar;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloud;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pubRangeImage;

  //  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubExtractedCloud;
  //  rclcpp::Publisher<Utils::CloudInfo>::SharedPtr pubLaserCloudInfo;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudBasic;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudUndistorted;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerCloud;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfaceCloud;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
  rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
  std::deque<sensor_msgs::msg::Imu> imuQueue;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom;
  rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;
  std::deque<nav_msgs::msg::Odometry> odomQueue;

  std::deque<sensor_msgs::msg::PointCloud2> cloudQueue;
  sensor_msgs::msg::PointCloud2 currentCloudMsg;

  //  pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;

  // Image Projection

  std::mutex imuLock;
  std::mutex odoLock;

  double * imuTime = new double[queueLength];
  double * imuRotX = new double[queueLength];
  double * imuRotY = new double[queueLength];
  double * imuRotZ = new double[queueLength];

  int imuPointerCur;
  bool firstPointFlag;
  Eigen::Affine3f transStartInverse;

  pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
  pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
  pcl::PointCloud<PointType>::Ptr fullCloud;
  pcl::PointCloud<PointType>::Ptr extractedCloud;
  pcl::PointCloud<PointType>::Ptr cloudDeskewed;

  int ringFlag = 0;
  int deskewFlag;
  cv::Mat rangeMat;
  cv::Mat HSV;

  bool odomDeskewFlag;
  float odomIncreX;
  float odomIncreY;
  float odomIncreZ;

  Utils::CloudInfo cloudInfo;
  double timeScanCur;
  double timeScanEnd;
  std_msgs::msg::Header cloudHeader;

  std::vector<int> columnIdnCountVec;

  void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imuMsg);
  void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odometryMsg);
  void cloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg);
  bool cachePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr & laserCloudMsg);
  bool deskewInfo();
  void imuDeskewInfo();
  void odomDeskewInfo();
  void publishImage();
  void publishClouds(std::string frame_name);
  void projectPointCloud();
  void findRotation(double pointTime, float * rotXCur, float * rotYCur, float * rotZCur);
  void findPosition(double relTime, float * posXCur, float * posYCur, float * posZCur);
  PointType deskewPoint(PointType * point, double relTime);
  void cloudExtraction();
  void resetParameters();
  void allocateMemory();


  // Feature Extraction

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerPoints;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfacePoints;

  pcl::PointCloud<PointType>::Ptr cornerCloud;
  pcl::PointCloud<PointType>::Ptr surfaceCloud;

  pcl::VoxelGrid<PointType> downSizeFilter;

  struct smoothness_t
  {
    float value;
    size_t ind;
  };
  struct by_value{
    bool operator()(smoothness_t const &left, smoothness_t const &right) {
      return left.value < right.value;
    }
  };

  std::vector<smoothness_t> cloudSmoothness;
  float * cloudCurvature;
  int * cloudNeighborPicked;
  int * cloudLabel;


  void initializationValue();
  void calculateSmoothness();
  void markOccludedPoints();
  void extractFeatures();
  void freeCloudInfoMemory();
//  void publishFeatureCloud();



};
}  // namespace loam_feature_localization

#endif  // LOAM_FEATURE_LOCALIZATION__LOAM_FEATURE_LOCALIZATION_HPP_
