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
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <boost/filesystem.hpp>

#include <cv_bridge/cv_bridge.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

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
  std::string output_odometry_frame_;
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

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubMapCorner;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubMapSurface;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudBasic;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudUndistorted;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerCloud;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfaceCloud;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryGlobal;

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


  // Feature Matching

    gtsam::NonlinearFactorGraph gtSAMgraph;
  gtsam::Values initialEstimate;
  gtsam::Values optimizedEstimate;
  gtsam::ISAM2 *isam;
  gtsam::Values isamCurrentEstimate;
  Eigen::MatrixXd poseCovariance;

  std::vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
  std::vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

  pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
  pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
  pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
  pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

  pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf feature set from odoOptimization

  pcl::PointCloud<PointType>::Ptr laserCloudOri;
  pcl::PointCloud<PointType>::Ptr coeffSel;

  std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
  std::vector<PointType> coeffSelCornerVec;
  std::vector<bool> laserCloudOriCornerFlag;
  std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
  std::vector<PointType> coeffSelSurfVec;
  std::vector<bool> laserCloudOriSurfFlag;

  std::map<int, std::pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;
  pcl::PointCloud<PointType>::Ptr map_corner;
  pcl::PointCloud<PointType>::Ptr map_surface;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

  pcl::VoxelGrid<PointType> downSizeFilterCorner;
  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  pcl::VoxelGrid<PointType> downSizeFilterICP;
  pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

  rclcpp::Time timeLaserInfoStamp;
  double timeLaserInfoCur;

  float transformTobeMapped[6];

  std::mutex mtx;
  std::mutex mtxLoopInfo;

  bool isDegenerate = false;
  Eigen::Matrix<float, 6, 6> matP;

  int laserCloudCornerFromMapDSNum = 0;
  int laserCloudSurfFromMapDSNum = 0;
  int laserCloudCornerLastDSNum = 0;
  int laserCloudSurfLastDSNum = 0;

  bool aLoopIsClosed = false;
  std::map<int, int> loopIndexContainer; // from new to old
  std::vector<std::pair<int, int>> loopIndexQueue;
  std::vector<gtsam::Pose3> loopPoseQueue;
  std::vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
  std::deque<std_msgs::msg::Float64MultiArray> loopInfoVec;

  nav_msgs::msg::Path globalPath;

  Eigen::Affine3f transPointAssociateToMap;
  Eigen::Affine3f incrementalOdometryAffineFront;
  Eigen::Affine3f incrementalOdometryAffineBack;

  std::unique_ptr<tf2_ros::TransformBroadcaster> br;

  void updateInitialGuess();
  void extractSurroundingKeyFrames();
  void extractNearby();
  void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract);
  pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn);
  void downsampleCurrentScan();
  void scan2MapOptimization();
  void cornerOptimization();
  void surfOptimization();
  void combineOptimizationCoeffs();
  bool LMOptimization(int iterCount);
  void transformUpdate();
  void updatePointAssociateToMap();
  Eigen::Affine3f trans2Affine3f(float transformIn[]);
  void pointAssociateToMap(PointType const * const pi, PointType * const po);
  float constraintTransformation(float value, float limit);
  void saveKeyFramesAndFactor();
  bool saveFrame();
  Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint);
  void updatePath(const PointTypePose& pose_in);
  void correctPoses();
  void publishOdometry();
  void addOdomFactor();
  gtsam::Pose3 trans2gtsamPose(float transformIn[]);
  gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint);

};
}  // namespace loam_feature_localization

#endif  // LOAM_FEATURE_LOCALIZATION__LOAM_FEATURE_LOCALIZATION_HPP_
