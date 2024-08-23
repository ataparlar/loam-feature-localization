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

#ifndef LOAM_FEATURE_LOCALIZATION__IMU_PREINTEGRATION_HPP_
#define LOAM_FEATURE_LOCALIZATION__IMU_PREINTEGRATION_HPP_

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

#include <boost/filesystem.hpp>

#include <cv_bridge/cv_bridge.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <deque>
#include <memory>
#include <string>

namespace loam_feature_localization
{
class TransformFusion
{
public:
  using SharedPtr = std::shared_ptr<TransformFusion>;
  using ConstSharedPtr = const std::shared_ptr<TransformFusion>;

  explicit TransformFusion(
    std::string base_link_frame, std::string lidar_frame, std::string odometry_frame);

  void lidar_odometry_handler(const nav_msgs::msg::Odometry::SharedPtr odom_msg);
  void imu_odometry_handler(
    const nav_msgs::msg::Odometry::SharedPtr odomMsg, rclcpp::Logger logger_,
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry,
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubImuPath);

private:
  Utils::SharedPtr utils;

  //  rclcpp::Logger logger;

  std::string base_link_frame_;
  std::string lidar_frame_;
  std::string odometry_frame_;

  std::mutex mtx;

  Eigen::Isometry3d lidarOdomAffine;
  Eigen::Isometry3d imuOdomAffineFront;
  Eigen::Isometry3d imuOdomAffineBack;

  std::shared_ptr<tf2_ros::Buffer> tfBuffer;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;
  std::shared_ptr<tf2_ros::TransformListener> tfListener;
  tf2::Stamped<tf2::Transform> lidar2Baselink;

  double lidarOdomTime = -1;
  std::deque<nav_msgs::msg::Odometry> imuOdomQueue;

  Eigen::Isometry3d odom2affine(nav_msgs::msg::Odometry odom);
};

class ImuPreintegration
{
public:
  using SharedPtr = std::shared_ptr<ImuPreintegration>;
  using ConstSharedPtr = const std::shared_ptr<ImuPreintegration>;

  explicit ImuPreintegration(
    std::string base_link_frame, std::string lidar_frame, std::string odometry_frame,
    float lidar_imu_x, float lidar_imu_y, float lidar_imu_z, float imu_gravity, float imu_acc_noise,
    float imu_acc_bias, float imu_gyro_noise, float imu_gyro_bias);

  void odometry_handler(const nav_msgs::msg::Odometry::SharedPtr odomMsg, rclcpp::Logger logger_);
  void imu_handler(
    const sensor_msgs::msg::Imu::SharedPtr imu_raw,
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry);

private:
  Utils::SharedPtr utils;

  std::mutex mtx;

  std::string base_link_frame_;
  std::string lidar_frame_;
  std::string odometry_frame_;
  float lidar_imu_x_;
  float lidar_imu_y_;
  float lidar_imu_z_;

  bool systemInitialized = false;
  nav_msgs::msg::Odometry last_odom = nav_msgs::msg::Odometry();

  gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
  gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
  gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
  gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
  gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
  gtsam::Vector noiseModelBetweenBias;

  gtsam::PreintegratedImuMeasurements * imuIntegratorOpt_;
  gtsam::PreintegratedImuMeasurements * imuIntegratorImu_;

  std::deque<sensor_msgs::msg::Imu> imuQueOpt;
  std::deque<sensor_msgs::msg::Imu> imuQueImu;

  gtsam::Pose3 prevPose_;
  gtsam::Vector3 prevVel_;
  gtsam::NavState prevState_;
  gtsam::imuBias::ConstantBias prevBias_;

  gtsam::NavState prevStateOdom;
  gtsam::imuBias::ConstantBias prevBiasOdom;

  bool doneFirstOpt = false;
  double lastImuT_imu = -1;
  double lastImuT_opt = -1;

  gtsam::ISAM2 optimizer;
  gtsam::NonlinearFactorGraph graphFactors;
  gtsam::Values graphValues;

  const double delta_t = 0;

  int key = 1;

  //  gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(),
  //  -extTrans.y(), -extTrans.z())); gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0,
  //  0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));
  gtsam::Pose3 imu2Lidar;
  gtsam::Pose3 lidar2Imu;

  void reset_optimization();
  void reset_params();
  bool failure_detection(
    const gtsam::Vector3 & velCur, const gtsam::imuBias::ConstantBias & biasCur,
    const rclcpp::Logger & logger_);
};

}  // namespace loam_feature_localization

#endif  // LOAM_FEATURE_LOCALIZATION__IMU_PREINTEGRATION_HPP_
