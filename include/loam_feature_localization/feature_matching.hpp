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

#ifndef LOAM_FEATURE_LOCALIZATION__FEATURE_MATCHING_HPP_
#define LOAM_FEATURE_LOCALIZATION__FEATURE_MATCHING_HPP_

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
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <tf2_ros/transform_broadcaster.h>

#include <deque>
#include <memory>
#include <string>

namespace loam_feature_localization
{
class FeatureMatching
{
public:
  using SharedPtr = std::shared_ptr<FeatureMatching>;
  using ConstSharedPtr = const std::shared_ptr<FeatureMatching>;

  explicit FeatureMatching(
    int n_scan, int horizon_scan, std::string corner_map_path, std::string surface_map_path,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_map_corner,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_map_surface,
    rclcpp::Time now, const Utils::SharedPtr & utils,
    const std::string & odometry_frame,  const std::string & base_link_frame,
    double surrounding_key_frame_search_radius, double surrounding_key_frame_adding_angle_threshold,
    double surrounding_key_frame_adding_dist_threshold, double surrounding_key_frame_density,
    double mapping_corner_leaf_size, double mapping_surface_leaf_size,
    int min_edge_feature_number, int min_surface_feature_number,
    double rotation_tollerance, double z_tollerance, double imu_rpy_weight);

  void laser_cloud_info_handler(
    const Utils::CloudInfo msg_in, std_msgs::msg::Header header,
    pcl::PointCloud<PointType>::Ptr cloud_corner, pcl::PointCloud<PointType>::Ptr cloud_surface,
    const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr & pub_odom_laser,
    const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr & pub_odom_laser_incremental,
    const std::unique_ptr<tf2_ros::TransformBroadcaster> & br,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_key_poses,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_recent_key_frames,
//    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_cloud_registered,
    const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr & pub_path);
  void publish_odometry(
    const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr & pub_odom_laser,
    const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr & pub_odom_laser_incremental,
    const std::unique_ptr<tf2_ros::TransformBroadcaster> & br);
  void publish_frames(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_key_poses,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_recent_key_frames,
//    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_cloud_registered,
    const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr & pub_path);


private:

  Utils::SharedPtr utils_;

  int n_scan_;
  int horizon_scan_;
  std::string map_corner_path_;
  std::string map_surface_path_;
  double surrounding_key_frame_search_radius_;
  double surrounding_key_frame_adding_angle_threshold_;
  double surrounding_key_frame_adding_dist_threshold_;
  double surrounding_key_frame_density_;
  double mapping_corner_leaf_size_;
  double mapping_surface_leaf_size_;
  int min_edge_feature_number_;
  int min_surface_feature_number_;
  double rotation_tollerance_;
  double z_tollerance_;
  double imu_rpy_weight_;
  std::string odometry_frame_;
  std::string base_link_frame_;

  // gtsam
  gtsam::NonlinearFactorGraph gtsam_graph_;
  gtsam::Values initial_estimate_;
  gtsam::Values optimized_estimate_;
  gtsam::ISAM2 *isam_;
  gtsam::Values isam_current_estimate_;
  Eigen::MatrixXd pose_covariance_;

  std::deque<nav_msgs::msg::Odometry> gps_queue_;
  Utils::CloudInfo cloud_info_;

  std::vector<pcl::PointCloud<PointType>::Ptr> corner_cloud_key_frames;
  std::vector<pcl::PointCloud<PointType>::Ptr> surface_cloud_key_frames;

  pcl::PointCloud<PointType>::Ptr cloud_key_poses_3d_;
  pcl::PointCloud<PointTypePose>::Ptr cloud_key_poses_6d_;
  pcl::PointCloud<PointType>::Ptr copy_cloud_key_poses_3d_;
  pcl::PointCloud<PointTypePose>::Ptr copy_cloud_key_poses_6d_;

  pcl::PointCloud<PointType>::Ptr laser_cloud_corner_last_; // corner feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laser_cloud_surface_last_; // surf feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laser_cloud_corner_last_ds_; // downsampled corner feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laser_cloud_surface_last_ds_; // downsampled surf feature set from odoOptimization

  pcl::PointCloud<PointType>::Ptr laser_cloud_ori_;
  pcl::PointCloud<PointType>::Ptr coeff_sel_;

  std::vector<PointType> laser_cloud_ori_corner_vec_; // corner point holder for parallel computation
  std::vector<PointType> coeff_sel_corner_vec_;
  std::vector<bool> laser_cloud_ori_corner_flag_;
  std::vector<PointType> laser_cloud_ori_surface_vec_; // surf point holder for parallel computation
  std::vector<PointType> coeff_sel_surface_vec_;
  std::vector<bool> laser_cloud_ori_surface_flag_;

  std::map<int, std::pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laser_cloud_map_container_;
  pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map_;
  pcl::PointCloud<PointType>::Ptr laser_cloud_surface_from_map_;
  pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map_ds_;
  pcl::PointCloud<PointType>::Ptr laser_cloud_surface_from_map_ds_;
  pcl::PointCloud<PointType>::Ptr map_corner_;
  pcl::PointCloud<PointType>::Ptr map_surface_;
  pcl::PointCloud<PointType>::Ptr map_corner_clipped_;
  pcl::PointCloud<PointType>::Ptr map_surface_clipped_;

  pcl::KdTreeFLANN<PointType>::Ptr kdtree_corner_from_map_;
  pcl::KdTreeFLANN<PointType>::Ptr kdtree_surface_from_map_;

  pcl::KdTreeFLANN<PointType>::Ptr kdtree_surrounding_key_poses_;
  pcl::KdTreeFLANN<PointType>::Ptr kdtree_history_key_poses_;

  pcl::octree::OctreePointCloudSearch<PointType>::Ptr map_corner_octree_;
  pcl::octree::OctreePointCloudSearch<PointType>::Ptr map_surface_octree_;
  pcl::VoxelGrid<PointType> down_size_filter_corner_;
  pcl::VoxelGrid<PointType> down_size_filter_surface_;
  pcl::VoxelGrid<PointType> down_size_filter_icp_;
  pcl::VoxelGrid<PointType> down_size_filter_surrounding_key_poses_; // for surrounding key poses of scan-to-map optimization

  rclcpp::Time time_laser_info_stamp_;
  double time_laser_info_cur_;

  float transform_to_be_mapped_[6];
  float transform_to_be_mapped_old_[3];

  std::mutex mtx_;
  std::mutex mtx_loop_info_;

  bool is_degenerate_ = false;
  Eigen::Matrix<float, 6, 6> mat_p_;

  int laser_cloud_corner_from_map_ds_num_ = 0;
  int laser_cloud_surface_from_map_ds_num_ = 0;
  int laser_cloud_corner_last_ds_num_ = 0;
  int laser_cloud_surface_last_ds_num_ = 0;

  bool a_loop_is_closed_ = false;
  std::map<int, int> loop_index_container_; // from new to old
  std::vector<std::pair<int, int>> loop_index_queue_;
  std::vector<gtsam::Pose3> loop_pose_queue;
  std::vector<gtsam::noiseModel::Diagonal::shared_ptr> loop_noise_queue_;
  std::deque<std_msgs::msg::Float64MultiArray> loop_info_vec_;

  nav_msgs::msg::Path global_path_;

  Eigen::Affine3f trans_point_associate_to_map_;
  Eigen::Affine3f incremental_odometry_affine_front_;
  Eigen::Affine3f incremental_odometry_affine_back_;


  void allocate_memory(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_map_corner,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_map_surface,
    rclcpp::Time now);
  void point_associate_to_map(PointType const * const pi, PointType * const po);
  pcl::PointCloud<PointType>::Ptr transform_point_cloud(pcl::PointCloud<PointType>::Ptr cloud_in, PointTypePose* transform_in);
  gtsam::Pose3 pcl_point_to_gtsam_pose3(PointTypePose this_point);
  gtsam::Pose3 trans_to_gtsam_pose(float transform_in[]);
  Eigen::Affine3f pcl_point_to_affine3f(PointTypePose this_point);
  Eigen::Affine3f trans_to_affine3f(float transform_in[]);
  PointTypePose trans_to_point_type_pose(float transform_in[]);
  void update_initial_guess();
  void extract_nearby();
  void extract_cloud();
  void extract_surrounding_key_frames();
  void downsample_current_scan();
  void update_point_associate_to_map();
  void corner_optimization();
  void surface_optimization();
  void combine_optimization_coeffs();
  bool lm_optimization(int iter_count);
  void scan_to_map_optimization();
  void transform_update();
  float constraint_transformation(float value, float limit);
  bool save_frame();
  void add_odom_factor();
  void add_gps_factor();
  void add_loop_factor();
  void save_key_frames_and_factor();
  void correct_poses();
  void update_path(const PointTypePose& pose_in);
  void publish_cloud(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub,
    pcl::PointCloud<PointType>::Ptr thisCloud, rclcpp::Time thisStamp,
    std::string thisFrame);




};

} // namespace loam_feature_localization


#endif  // LOAM_FEATURE_LOCALIZATION__FEATURE_MATCHING_HPP_
