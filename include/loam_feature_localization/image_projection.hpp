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

#ifndef LOAM_FEATURE_LOCALIZATION__IMAGE_PROJECTION_HPP_
#define LOAM_FEATURE_LOCALIZATION__IMAGE_PROJECTION_HPP_

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
class ImageProjection
{
public:
  using SharedPtr = std::shared_ptr<ImageProjection>;
  using ConstSharedPtr = const std::shared_ptr<ImageProjection>;

  explicit ImageProjection(int N_SCAN, int Horizon_SCAN);

private:
  std::mutex imu_lock_;
  std::mutex odom_lock_;

  Utils::SharedPtr utils_;

  int n_scan_;
  int horizon_scan_;

  std::deque<sensor_msgs::msg::Imu> imu_queue_;
  std::deque<nav_msgs::msg::Odometry> odom_queue_;
  std::deque<pcl::PointCloud<PointType>> cloud_queue_;

  pcl::PointCloud<PointType> current_cloud_msg_;

  double * imu_time_ = new double[queueLength];
  double * imu_rot_x_ = new double[queueLength];
  double * imu_rot_y_ = new double[queueLength];
  double * imu_rot_z_ = new double[queueLength];

  int imu_pointer_cur_;
  bool first_point_flag_;
  Eigen::Affine3f trans_start_inverse_;

  pcl::PointCloud<PointXYZIRT>::Ptr laser_cloud_in_;
  //  pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
  pcl::PointCloud<PointType>::Ptr full_cloud_;
  pcl::PointCloud<PointType>::Ptr extracted_cloud_;

  int ring_flag_ = 0;
  int deskew_flag_;
  cv::Mat range_mat_;

  bool odom_deskew_flag_;
  float odom_incre_x_;
  float odom_incre_y_;
  float odom_incre_z_;

  Utils::CloudInfo cloud_info_;
  double time_scan_cur_;
  double time_scan_end_;
  std_msgs::msg::Header cloud_header_;

  std::vector<int> column_ind_count_vec_;

  void allocate_memory();
  void reset_parameters();
  void imu_handler(const sensor_msgs::msg::Imu::SharedPtr imuMsg);
  void odometry_handler(const nav_msgs::msg::Odometry::SharedPtr odometryMsg);
  void cloud_handler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg);
  bool cache_point_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr & laserCloudMsg);
  bool deskew_info();
  void imu_deskew_info();
  void odom_deskew_info();
  void find_rotation(double pointTime, float * rotXCur, float * rotYCur, float * rotZCur);
  void find_position(double relTime, float * posXCur, float * posYCur, float * posZCur);
  PointType deskew_point(PointType * point, double relTime);
  void project_point_cloud();
  void cloud_extraction();
  void publish_clouds();

};
} // namespace loam_feature_localization


#endif  // LOAM_FEATURE_LOCALIZATION__IMAGE_PROJECTION_HPP_
