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

#include "loam_feature_localization/loam_feature_localization.hpp"

#include "cv_bridge/cv_bridge.h"
#include "loam_feature_localization/utils.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>


namespace loam_feature_localization
{
LoamFeatureLocalization::LoamFeatureLocalization(const rclcpp::NodeOptions & options)
: Node("loam_feature_localization", options)
{
  this->declare_parameter("imu_topic", "");
  this->declare_parameter("odom_topic", "");
  this->declare_parameter("point_cloud_topic", "");
  this->declare_parameter("output_odometry_frame", "");
  this->declare_parameter("corner_map_path", "");
  this->declare_parameter("surface_map_path", "");
  this->declare_parameter("lidar_imu_x", 0.0);
  this->declare_parameter("lidar_imu_y", 0.0);
  this->declare_parameter("lidar_imu_z", 0.0);
  this->declare_parameter("lidar_imu_roll", 0.0);
  this->declare_parameter("lidar_imu_pitch", 0.0);
  this->declare_parameter("lidar_imu_yaw", 0.0);
  this->declare_parameter("lidar_min_range", 1.0);
  this->declare_parameter("lidar_max_range", 120.0);
  this->declare_parameter("N_SCAN", 32);
  this->declare_parameter("Horizon_SCAN", 2000);
  this->declare_parameter("edge_threshold", 120.0);
  this->declare_parameter("surface_threshold", 120.0);

  this->declare_parameter("odometry_surface_leaf_size", 0.4);
  this->declare_parameter("mapping_corner_leaf_size", 0.2);
  this->declare_parameter("mapping_surf_leaf_size", 0.4);
  this->declare_parameter("surrounding_key_frame_size", 50.0);
  this->declare_parameter("edge_feature_min_valid_num", 10.0);
  this->declare_parameter("surf_feature_min_valid_num", 100.0);
  this->declare_parameter("imu_rpy_weight", 0.01);
  this->declare_parameter("rotation_tollerance", 1000.0);
  this->declare_parameter("z_tollerance", 1000.0);
  this->declare_parameter("surrounding_key_frame_adding_angle_threshold", 1.0);
  this->declare_parameter("surrounding_key_frame_adding_dist_threshold", 0.2);
  this->declare_parameter("surrounding_key_frame_density", 2.0);
  this->declare_parameter("surrounding_key_frame_search_radius", 2.0);

  imu_topic_ = this->get_parameter("imu_topic").as_string();
  odom_topic_ = this->get_parameter("odom_topic").as_string();
  point_cloud_topic_ = this->get_parameter("point_cloud_topic").as_string();
  output_odometry_frame_ = this->get_parameter("point_cloud_topic").as_string();
  corner_map_path_ = this->get_parameter("corner_map_path").as_string();
  surface_map_path_ = this->get_parameter("surface_map_path").as_string();
  lidar_imu_x_ = this->get_parameter("lidar_imu_x").as_double();
  lidar_imu_y_ = this->get_parameter("lidar_imu_y").as_double();
  lidar_imu_z_ = this->get_parameter("lidar_imu_z").as_double();
  lidar_imu_roll_ = this->get_parameter("lidar_imu_roll").as_double();
  lidar_imu_pitch_ = this->get_parameter("lidar_imu_pitch").as_double();
  lidar_imu_yaw_ = this->get_parameter("lidar_imu_yaw").as_double();
  lidar_min_range_ = this->get_parameter("lidar_min_range").as_double();
  lidar_max_range_ = this->get_parameter("lidar_max_range").as_double();
  N_SCAN_ = this->get_parameter("N_SCAN").as_int();
  Horizon_SCAN_ = this->get_parameter("Horizon_SCAN").as_int();
  edge_threshold_ = this->get_parameter("edge_threshold").as_double();
  surface_threshold_ = this->get_parameter("surface_threshold").as_double();

  odometry_surface_leaf_size_ = this->get_parameter("odometry_surface_leaf_size").as_double();
  mapping_corner_leaf_size_ = this->get_parameter("mapping_corner_leaf_size").as_double();
  mapping_surf_leaf_size_ = this->get_parameter("mapping_surf_leaf_size").as_double();
  surrounding_key_frame_size_ = this->get_parameter("surrounding_key_frame_size").as_double();
  edge_feature_min_valid_num_ = this->get_parameter("edge_feature_min_valid_num").as_double();
  surf_feature_min_valid_num_ = this->get_parameter("surf_feature_min_valid_num").as_double();
  imu_rpy_weight_ = this->get_parameter("imu_rpy_weight").as_double();
  rotation_tollerance_ = this->get_parameter("rotation_tollerance").as_double();
  z_tollerance_ = this->get_parameter("z_tollerance").as_double();
  surrounding_key_frame_adding_angle_threshold_ =
    this->get_parameter("surrounding_key_frame_adding_angle_threshold").as_double();
  surrounding_key_frame_adding_dist_threshold_ =
    this->get_parameter("surrounding_key_frame_adding_dist_threshold").as_double();
  surrounding_key_frame_density_ = this->get_parameter("surrounding_key_frame_density").as_double();
  surrounding_key_frame_search_radius_ =
    this->get_parameter("surrounding_key_frame_search_radius").as_double();

  callbackGroupLidar = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  callbackGroupImu = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  callbackGroupOdom = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  auto lidarOpt = rclcpp::SubscriptionOptions();
  lidarOpt.callback_group = callbackGroupLidar;
  auto imuOpt = rclcpp::SubscriptionOptions();
  imuOpt.callback_group = callbackGroupImu;
  auto odomOpt = rclcpp::SubscriptionOptions();
  odomOpt.callback_group = callbackGroupOdom;

  subImu = create_subscription<sensor_msgs::msg::Imu>(
    imu_topic_, rclcpp::SensorDataQoS(),
    std::bind(&LoamFeatureLocalization::imu_handler, this, std::placeholders::_1), imuOpt);
  subImuOdom = create_subscription<nav_msgs::msg::Odometry>(
    "/odometry_incremental", rclcpp::SensorDataQoS(),
    std::bind(&LoamFeatureLocalization::imu_odometry_handler, this, std::placeholders::_1), odomOpt);
  subLaserOdom = create_subscription<nav_msgs::msg::Odometry>(
    "/odometry_matched", rclcpp::SensorDataQoS(),
    std::bind(&LoamFeatureLocalization::laser_odometry_handler, this, std::placeholders::_1), odomOpt);
  subLaserCloud = create_subscription<sensor_msgs::msg::PointCloud2>(
    point_cloud_topic_, rclcpp::SensorDataQoS(),
    std::bind(&LoamFeatureLocalization::cloud_handler, this, std::placeholders::_1), lidarOpt);

  pubRangeImage =
    create_publisher<sensor_msgs::msg::Image>("/range_image", rclcpp::SensorDataQoS());
  pubMapCorner =
    create_publisher<sensor_msgs::msg::PointCloud2>("/map_corner", rclcpp::SensorDataQoS());
  pubMapSurface =
    create_publisher<sensor_msgs::msg::PointCloud2>("/map_surface", rclcpp::SensorDataQoS());
  pubCloudBasic =
    create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_basic", rclcpp::SensorDataQoS());
  pubCloudUndistorted =
    create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_undistorted", rclcpp::SensorDataQoS());
  pubCornerCloud =
    create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_corner", rclcpp::SensorDataQoS());
  pubSurfaceCloud =
    create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_surface", rclcpp::SensorDataQoS());
  pubImuOdom =
    create_publisher<nav_msgs::msg::Odometry>("/imu_odom", rclcpp::SensorDataQoS());
  pubLaserOdometryGlobal =
    create_publisher<nav_msgs::msg::Odometry>("/laser_odom", rclcpp::SensorDataQoS());
  pubImuPath =
    create_publisher<nav_msgs::msg::Path>("/imu_odom", rclcpp::SensorDataQoS());

  float imu_gravity = 0.0;
  float imu_acc_noise = 0.0;
  float imu_acc_bias = 0.0;
  float imu_gyro_noise = 0.0;
  float imu_gyro_bias = 0.0;

  // Objects
  utils_ = std::make_shared<Utils>();
  imu_preintegration_ = std::make_shared<ImuPreintegration>(
    "base_link", "lidar_link", "odom_link",
    lidar_imu_x_, lidar_imu_y_, lidar_imu_z_,
    imu_gravity,
    imu_acc_noise, imu_acc_bias, imu_gyro_noise, imu_gyro_bias);
  transform_fusion_ = std::make_shared<TransformFusion>("base_link", "lidar_link", "odom_link");



}

void LoamFeatureLocalization::imu_handler(const sensor_msgs::msg::Imu::SharedPtr imu_msg)
{
  imu_preintegration_->imu_handler(imu_msg, pubImuOdom);


}

void LoamFeatureLocalization::imu_odometry_handler(const nav_msgs::msg::Odometry::SharedPtr odom_msg)
{
  imu_preintegration_->odometry_handler(odom_msg, this->get_logger());
  transform_fusion_->imu_odometry_handler(odom_msg, this->get_logger(), pubImuOdom, pubImuPath);

}

void LoamFeatureLocalization::cloud_handler(const sensor_msgs::msg::PointCloud2::SharedPtr laser_cloud_msg)
{

}

}  // namespace loam_feature_localization

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(loam_feature_localization::LoamFeatureLocalization)
