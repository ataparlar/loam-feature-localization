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

#ifndef LOAM_FEATURE_LOCALIZATION__FEATURE_EXTRACTION_HPP_
#define LOAM_FEATURE_LOCALIZATION__FEATURE_EXTRACTION_HPP_

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
#include <tf2_ros/transform_broadcaster.h>

#include <deque>
#include <memory>
#include <string>

namespace loam_feature_localization
{
class FeatureExtraction
{
public:
  using SharedPtr = std::shared_ptr<FeatureExtraction>;
  using ConstSharedPtr = const std::shared_ptr<FeatureExtraction>;

  explicit FeatureExtraction( const Utils::SharedPtr & utils,
    int N_SCAN, int Horizon_SCAN, double odometry_surface_leaf_size, double edge_threshold,
    double surface_threshold, std::string lidar_frame);

  void laser_cloud_info_handler(
    const Utils::CloudInfo & msg_in, const std_msgs::msg::Header & cloud_header,
    const pcl::PointCloud<PointType>::Ptr & extracted_cloud);
  void publish_feature_cloud(
    const rclcpp::Time & now,
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerCloud,
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfaceCloud);

private:
  Utils::SharedPtr utils_;

  int n_scan_;
  int horizon_scan_;
  double odometry_surface_leaf_size_;
  double edge_threshold_;
  double surface_threshold_;
  std::string lidar_frame_;

    struct SmoothnessT
  {
    float value;
    size_t ind;
  };

  struct ByValue
  {
    bool operator()(SmoothnessT const & left, SmoothnessT const & right)
    {
      return left.value < right.value;
    }
  };

  Utils::CloudInfo cloud_info_;

  pcl::VoxelGrid<PointType> down_size_filter_;

  pcl::PointCloud<PointType>::Ptr cloud_deskewed_;
  pcl::PointCloud<PointType>::Ptr extracted_cloud_;
  pcl::PointCloud<PointType>::Ptr corner_cloud_;
  pcl::PointCloud<PointType>::Ptr surface_cloud_;

  std_msgs::msg::Header cloud_header_;

  std::vector<SmoothnessT> cloud_smoothness_;
  float * cloud_curvature_;
  int * cloud_neighbor_picked_;
  int * cloud_label_;

  void initialization_value();
  void calculate_smoothness();
  void mark_occluded_points();
  void extract_features();
  void free_cloud_info_memory();
};
}  // namespace loam_feature_localization

#endif  // LOAM_FEATURE_LOCALIZATION__FEATURE_EXTRACTION_HPP_
