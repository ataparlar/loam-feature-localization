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
#include "imu_preintegration.hpp"
#include "image_projection.hpp"
#include "feature_extraction.hpp"

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

  Utils::SharedPtr utils_;
  ImuPreintegration::SharedPtr imu_preintegration_;
  TransformFusion::SharedPtr transform_fusion_;
  ImageProjection::SharedPtr image_projection_;
  FeatureExtraction::SharedPtr feature_extraction_;


  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud;
  rclcpp::CallbackGroup::SharedPtr callbackGroupLidar;
//  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloud;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_range_matrix;

  //  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubExtractedCloud;
  //  rclcpp::Publisher<Utils::CloudInfo>::SharedPtr pubLaserCloudInfo;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_corner;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_surface;
//  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudBasic;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_deskewed;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_corner;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_surface;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_imu;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_laser;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_imu;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_laser;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
  rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
  std::deque<sensor_msgs::msg::Imu> imu_queue;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_imu;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_laser;
  rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;

  void imu_handler(const sensor_msgs::msg::Imu::SharedPtr imu_msg);
  void imu_odometry_handler(const nav_msgs::msg::Odometry::SharedPtr odom_msg);
  void laser_odometry_handler(const nav_msgs::msg::Odometry::SharedPtr odom_msg);
  void cloud_handler(const sensor_msgs::msg::PointCloud2::SharedPtr laser_cloud_msg);


  // Feature Extraction

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerPoints;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfacePoints;

  pcl::PointCloud<PointType>::Ptr cornerCloud;
  pcl::PointCloud<PointType>::Ptr surfaceCloud;



};
}  // namespace loam_feature_localization

#endif  // LOAM_FEATURE_LOCALIZATION__LOAM_FEATURE_LOCALIZATION_HPP_
