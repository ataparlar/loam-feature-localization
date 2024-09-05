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

//  this->declare_parameter("lidar_imu_x", 0.0);
//  this->declare_parameter("lidar_imu_y", 0.0);
//  this->declare_parameter("lidar_imu_z", 0.0);
//  this->declare_parameter("lidar_imu_roll", 0.0);
//  this->declare_parameter("lidar_imu_pitch", 0.0);
//  this->declare_parameter("lidar_imu_yaw", 0.0);

  double ida[] = { 1.0,  0.0,  0.0,
                  0.0,  1.0,  0.0,
                  0.0,  0.0,  1.0};
  std::vector < double > id(ida, std::end(ida));
  double zea[] = {0.0, 0.0, 0.0};
  std::vector < double > ze(zea, std::end(zea));
  declare_parameter("extrinsic_rot", id);
  declare_parameter("extrinsic_rpy", id);
  declare_parameter("extrinsic_trans", ze);

  this->declare_parameter("lidar_min_range", 1.0);
  this->declare_parameter("lidar_max_range", 120.0);
  this->declare_parameter("N_SCAN", 32);
  this->declare_parameter("Horizon_SCAN", 2000);
  this->declare_parameter("edge_threshold", 120.0);
  this->declare_parameter("surface_threshold", 120.0);

  this->declare_parameter("imu_gravity", 0.0);
  this->declare_parameter("imu_acc_noise", 3.9939570888238808e-03);
  this->declare_parameter("imu_acc_bias", 6.4356659353532566e-05);
  this->declare_parameter("imu_gyro_noise", 1.5636343949698187e-03);
  this->declare_parameter("imu_gyro_bias", 3.5640318696367613e-05);

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

//  lidar_imu_x_ = this->get_parameter("lidar_imu_x").as_double();
//  lidar_imu_y_ = this->get_parameter("lidar_imu_y").as_double();
//  lidar_imu_z_ = this->get_parameter("lidar_imu_z").as_double();
//  lidar_imu_roll_ = this->get_parameter("lidar_imu_roll").as_double();
//  lidar_imu_pitch_ = this->get_parameter("lidar_imu_pitch").as_double();
//  lidar_imu_yaw_ = this->get_parameter("lidar_imu_yaw").as_double();

  get_parameter("extrinsic_rot", ext_rot_v_);
  get_parameter("extrinsic_rpy", ext_rpy_v_);
  get_parameter("extrinsic_trans", ext_trans_v_);


  lidar_min_range_ = this->get_parameter("lidar_min_range").as_double();
  lidar_max_range_ = this->get_parameter("lidar_max_range").as_double();
  N_SCAN_ = this->get_parameter("N_SCAN").as_int();
  Horizon_SCAN_ = this->get_parameter("Horizon_SCAN").as_int();
  edge_threshold_ = this->get_parameter("edge_threshold").as_double();
  surface_threshold_ = this->get_parameter("surface_threshold").as_double();

  imu_gravity_ = this->get_parameter("imu_gravity").as_double();
  imu_acc_noise_ = this->get_parameter("imu_acc_noise").as_double();
  imu_acc_bias_ = this->get_parameter("imu_acc_bias").as_double();
  imu_gyro_noise_ = this->get_parameter("imu_gyro_noise").as_double();
  imu_gyro_bias_ = this->get_parameter("imu_gyro_bias").as_double();

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

  auto lidar_opt = rclcpp::SubscriptionOptions();
  lidar_opt.callback_group = callbackGroupLidar;
  auto imu_opt = rclcpp::SubscriptionOptions();
  imu_opt.callback_group = callbackGroupImu;
  auto odom_opt = rclcpp::SubscriptionOptions();
  odom_opt.callback_group = callbackGroupOdom;

  sub_imu = create_subscription<sensor_msgs::msg::Imu>(
    imu_topic_, rclcpp::SensorDataQoS(),
    std::bind(&LoamFeatureLocalization::imu_handler, this, std::placeholders::_1), imu_opt);
  sub_odom_imu = create_subscription<nav_msgs::msg::Odometry>(
    "/odometry_incremental", rclcpp::SensorDataQoS(),
    std::bind(&LoamFeatureLocalization::imu_odometry_handler, this, std::placeholders::_1), odom_opt);
  sub_odom_laser = create_subscription<nav_msgs::msg::Odometry>(
    "/odometry_matched", rclcpp::SensorDataQoS(),
    std::bind(&LoamFeatureLocalization::laser_odometry_handler, this, std::placeholders::_1), odom_opt);
  sub_cloud = create_subscription<sensor_msgs::msg::PointCloud2>(
    point_cloud_topic_, rclcpp::SensorDataQoS(),
    std::bind(&LoamFeatureLocalization::cloud_handler, this, std::placeholders::_1), lidar_opt);

  std::cout << "point_cloud_topic_: " << point_cloud_topic_ << std::endl;

  pub_range_matrix =
    create_publisher<sensor_msgs::msg::Image>("/range_image", rclcpp::SensorDataQoS());
  pub_map_corner =
    create_publisher<sensor_msgs::msg::PointCloud2>("/map_corner", rclcpp::SensorDataQoS());
  pub_map_surface =
    create_publisher<sensor_msgs::msg::PointCloud2>("/map_surface", rclcpp::SensorDataQoS());
//  pubCloudBasic =
//    create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_basic", rclcpp::SensorDataQoS());
  pub_cloud_deskewed =
    create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_undistorted", rclcpp::SensorDataQoS());
  pub_cloud_corner =
    create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_corner", rclcpp::SensorDataQoS());
  pub_cloud_surface =
    create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_surface", rclcpp::SensorDataQoS());
  pub_key_poses_ =
    create_publisher<sensor_msgs::msg::PointCloud2>("/pub_key_poses", rclcpp::SensorDataQoS());
  pub_recent_key_frames_ =
    create_publisher<sensor_msgs::msg::PointCloud2>("/pub_recent_key_frames", rclcpp::SensorDataQoS());
  pub_cloud_registered_ =
    create_publisher<sensor_msgs::msg::PointCloud2>("/pub_cloud_registered", rclcpp::SensorDataQoS());
  pub_odom_imu =
    create_publisher<nav_msgs::msg::Odometry>("/imu_odom", rclcpp::SensorDataQoS());
  pub_odom_laser =
    create_publisher<nav_msgs::msg::Odometry>("/laser_odom", rclcpp::SensorDataQoS());
  pub_odom_laser_incremental =
    create_publisher<nav_msgs::msg::Odometry>("/laser_odom_incremental", rclcpp::SensorDataQoS());
  pub_path_imu =
    create_publisher<nav_msgs::msg::Path>("/imu_path", rclcpp::SensorDataQoS());
  pub_path_laser =
    create_publisher<nav_msgs::msg::Path>("/laser_path", rclcpp::SensorDataQoS());

  laser_tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

//  float imu_gravity = 0.0;
//  float imu_acc_noise = 0.0;
//  float imu_acc_bias = 0.0;
//  float imu_gyro_noise = 0.0;
//  float imu_gyro_bias = 0.0;

  ext_rot_ = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(ext_rot_v_.data(), 3, 3);
  ext_rpy_ = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(ext_rpy_v_.data(), 3, 3);
  ext_trans_ = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(ext_trans_v_.data(), 3, 1);
  ext_qrpy_ = Eigen::Quaterniond(ext_rpy_);

  // Objects
  utils_ = std::make_shared<Utils>(ext_rot_, ext_rpy_, ext_trans_);
  imu_preintegration_ = std::make_shared<ImuPreintegration>( utils_,
    "base_link", "lidar_link", "odom_link",
    ext_trans_[0], ext_trans_[1], ext_trans_[2],
    imu_gravity_,
    imu_acc_noise_, imu_acc_bias_, imu_gyro_noise_, imu_gyro_bias_);
  transform_fusion_ = std::make_shared<TransformFusion>( utils_,
    "base_link", "lidar_link", "odom_link");
  image_projection_ = std::make_shared<ImageProjection>( utils_,
    N_SCAN_, Horizon_SCAN_,
    120.0, 0.0, "lidar_link");
  feature_extraction_ = std::make_shared<FeatureExtraction>(utils_,
    N_SCAN_, Horizon_SCAN_,
    odometry_surface_leaf_size_,
    edge_threshold_, surface_threshold_,
    "lidar_link");
  feature_matching_ = std::make_shared<FeatureMatching>(
    N_SCAN_, Horizon_SCAN_,
    corner_map_path_, surface_map_path_, pub_map_corner, pub_map_surface,
    this->get_clock()->now(),
    utils_, surrounding_key_frame_search_radius_, surrounding_key_frame_adding_angle_threshold_,
    surrounding_key_frame_adding_dist_threshold_, surrounding_key_frame_density_,
    mapping_corner_leaf_size_, mapping_surf_leaf_size_,
    edge_feature_min_valid_num_, surf_feature_min_valid_num_,
    rotation_tollerance_, z_tollerance_, imu_rpy_weight_);
}

void LoamFeatureLocalization::imu_handler(const sensor_msgs::msg::Imu::SharedPtr imu_msg)
{
  imu_preintegration_->imu_handler(imu_msg, pub_odom_imu);
  image_projection_->imu_handler(imu_msg);
}

void LoamFeatureLocalization::imu_odometry_handler(const nav_msgs::msg::Odometry::SharedPtr odom_msg)
{
  imu_preintegration_->odometry_handler(odom_msg, this->get_logger());
  transform_fusion_->imu_odometry_handler(odom_msg, this->get_logger(), pub_odom_imu, pub_path_imu);
}



void LoamFeatureLocalization::cloud_handler(const sensor_msgs::msg::PointCloud2::SharedPtr laser_cloud_msg)
{
  pcl::PointCloud<PointType>::Ptr input_for_extraction;
  input_for_extraction.reset(new pcl::PointCloud<PointType>());

  image_projection_->cloud_handler(
    laser_cloud_msg, this->get_logger(), this->get_clock()->now(), pub_cloud_deskewed, input_for_extraction);

//  auto image = prepare_visualization_image(image_projection_->range_mat_);
  pub_range_matrix->publish(image_projection_->range_mat_for_vis_);

  feature_extraction_->laser_cloud_info_handler(
    image_projection_->cloud_info, laser_cloud_msg->header, image_projection_->extracted_cloud_to_pub_);
  feature_extraction_->publish_feature_cloud(this->get_clock()->now(), pub_cloud_corner, pub_cloud_surface);

  feature_matching_->laser_cloud_info_handler(
    image_projection_->cloud_info, laser_cloud_msg->header,
    feature_extraction_->corner_cloud_, feature_extraction_->surface_cloud_,
    pub_odom_laser, pub_odom_laser_incremental, laser_tf_, pub_key_poses_,
    pub_recent_key_frames_, pub_cloud_registered_, pub_path_laser);



}
void LoamFeatureLocalization::laser_odometry_handler(const nav_msgs::msg::Odometry::SharedPtr odom_msg)
{
  transform_fusion_->lidar_odometry_handler(odom_msg);
}



}  // namespace loam_feature_localization

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(loam_feature_localization::LoamFeatureLocalization)
