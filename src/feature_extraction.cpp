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

#include "loam_feature_localization/feature_extraction.hpp"

#include "pcl_conversions/pcl_conversions.h"

namespace loam_feature_localization
{
FeatureExtraction::FeatureExtraction(
  const Utils::SharedPtr & utils,
  int N_SCAN, int Horizon_SCAN,
  double odometry_surface_leaf_size,
  double edge_threshold, double surface_threshold,
  std::string lidar_frame)
{
  odometry_surface_leaf_size_ = odometry_surface_leaf_size;
  n_scan_ = N_SCAN;
  horizon_scan_ = Horizon_SCAN;
  edge_threshold_ = edge_threshold;
  surface_threshold_ = surface_threshold;
  lidar_frame_ = lidar_frame;
  utils_ = utils;

  initialization_value();
}

void FeatureExtraction::initialization_value()
{
  cloud_smoothness_.resize(n_scan_*horizon_scan_);

  down_size_filter_.setLeafSize(
    odometry_surface_leaf_size_, odometry_surface_leaf_size_, odometry_surface_leaf_size_);

  cloud_deskewed_.reset(new pcl::PointCloud<PointType>());
  extracted_cloud_.reset(new pcl::PointCloud<PointType>());
  corner_cloud_.reset(new pcl::PointCloud<PointType>());
  surface_cloud_.reset(new pcl::PointCloud<PointType>());

  cloud_curvature_ = new float[n_scan_*horizon_scan_];
  cloud_neighbor_picked_ = new int[n_scan_*horizon_scan_];
  cloud_label_ = new int[n_scan_*horizon_scan_];
}

void FeatureExtraction::laser_cloud_info_handler(
  const Utils::CloudInfo & msg_in, const std_msgs::msg::Header & cloud_header,
  const pcl::PointCloud<PointType>::Ptr & extracted_cloud)
{
  cloud_info_ = msg_in; // new cloud info
  cloud_header_ = cloud_header; // new cloud header
//  pcl::fromROSMsg(cloud_deskewed, *extracted_cloud_); // new cloud for extraction

  extracted_cloud_ = extracted_cloud;

  calculate_smoothness();

  mark_occluded_points();

  extract_features();

//  std::cout << "cloud_corner_: " << corner_cloud_->size() << std::endl;

//  publish_feature_cloud();
}

void FeatureExtraction::calculate_smoothness()
{
  int cloud_size = extracted_cloud_->points.size();
  std::cout << "cloud_size:  " << cloud_size << std::endl;
  for (int i = 5; i < cloud_size - 5; i++)
  {
    float diffRange = cloud_info_.point_range[i-5] + cloud_info_.point_range[i-4]
                      + cloud_info_.point_range[i-3] + cloud_info_.point_range[i-2]
                      + cloud_info_.point_range[i-1] - cloud_info_.point_range[i] * 10
                      + cloud_info_.point_range[i+1] + cloud_info_.point_range[i+2]
                      + cloud_info_.point_range[i+3] + cloud_info_.point_range[i+4]
                      + cloud_info_.point_range[i+5];

    cloud_curvature_[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;

    cloud_neighbor_picked_[i] = 0;
    cloud_label_[i] = 0;
    // cloudSmoothness for sorting
    cloud_smoothness_[i].value = cloud_curvature_[i];
    cloud_smoothness_[i].ind = i;
  }
}

void FeatureExtraction::mark_occluded_points()
{
  int cloud_size = extracted_cloud_->points.size();
  // mark occluded points and parallel beam points
  for (int i = 5; i < cloud_size - 6; ++i)
  {
    // occluded points
    float depth1 = cloud_info_.point_range[i];
    float depth2 = cloud_info_.point_range[i+1];
    int columnDiff = std::abs(int(cloud_info_.point_col_index[i+1] - cloud_info_.point_col_index[i]));
    if (columnDiff < 10){
      // 10 pixel diff in range image
      if (depth1 - depth2 > 0.3){
        cloud_neighbor_picked_[i - 5] = 1;
        cloud_neighbor_picked_[i - 4] = 1;
        cloud_neighbor_picked_[i - 3] = 1;
        cloud_neighbor_picked_[i - 2] = 1;
        cloud_neighbor_picked_[i - 1] = 1;
        cloud_neighbor_picked_[i] = 1;
      }else if (depth2 - depth1 > 0.3){
        cloud_neighbor_picked_[i + 1] = 1;
        cloud_neighbor_picked_[i + 2] = 1;
        cloud_neighbor_picked_[i + 3] = 1;
        cloud_neighbor_picked_[i + 4] = 1;
        cloud_neighbor_picked_[i + 5] = 1;
        cloud_neighbor_picked_[i + 6] = 1;
      }
    }
    // parallel beam
    float diff1 = std::abs(float(cloud_info_.point_range[i-1] - cloud_info_.point_range[i]));
    float diff2 = std::abs(float(cloud_info_.point_range[i+1] - cloud_info_.point_range[i]));

    if (diff1 > 0.02 * cloud_info_.point_range[i] && diff2 > 0.02 * cloud_info_.point_range[i])
      cloud_neighbor_picked_[i] = 1;
  }
}

void FeatureExtraction::extract_features()
{
  corner_cloud_->clear();
  surface_cloud_->clear();

  pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

  for (int i = 0; i < n_scan_; i++)
  {
    surfaceCloudScan->clear();

    for (int j = 0; j < 6; j++)
    {

      int sp = (cloud_info_.start_ring_index[i] * (6 - j) + cloud_info_.end_ring_index[i] * j) / 6;
      int ep = (cloud_info_.start_ring_index[i] * (5 - j) + cloud_info_.end_ring_index[i] * (j + 1)) / 6 - 1;

//      if (sp >= ep)
//        continue;
      if (sp >= cloud_smoothness_.size() || ep > cloud_smoothness_.size() || sp > ep) {
        std::cerr << "Invalid range: sp=" << sp << ", ep=" << ep << std::endl;
        continue;
      }


      std::sort(cloud_smoothness_.begin()+sp, cloud_smoothness_.begin()+ep, ByValue());

      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--)
      {
        int ind = cloud_smoothness_[k].ind;
        if (cloud_neighbor_picked_[ind] == 0 && cloud_curvature_[ind] > edge_threshold_)
        {
          largestPickedNum++;
          if (largestPickedNum <= 20){
            cloud_label_[ind] = 1;
            corner_cloud_->push_back(extracted_cloud_->points[ind]);
          } else {
            break;
          }

          cloud_neighbor_picked_[ind] = 1;
          for (int l = 1; l <= 5; l++)
          {
            int columnDiff = std::abs(int(cloud_info_.point_col_index[ind + l] - cloud_info_.point_col_index[ind + l - 1]));
            if (columnDiff > 10)
              break;
            cloud_neighbor_picked_[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--)
          {
            int columnDiff = std::abs(int(cloud_info_.point_col_index[ind + l] - cloud_info_.point_col_index[ind + l + 1]));
            if (columnDiff > 10)
              break;
            cloud_neighbor_picked_[ind + l] = 1;
          }
        }
      }

      for (int k = sp; k <= ep; k++)
      {
        int ind = cloud_smoothness_[k].ind;
        if (cloud_neighbor_picked_[ind] == 0 && cloud_curvature_[ind] < surface_threshold_)
        {

          cloud_label_[ind] = -1;
          cloud_neighbor_picked_[ind] = 1;

          for (int l = 1; l <= 5; l++) {
            int columnDiff = std::abs(int(cloud_info_.point_col_index[ind + l] - cloud_info_.point_col_index[ind + l - 1]));
            if (columnDiff > 10)
              break;

            cloud_neighbor_picked_[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            int columnDiff = std::abs(int(cloud_info_.point_col_index[ind + l] - cloud_info_.point_col_index[ind + l + 1]));
            if (columnDiff > 10)
              break;

            cloud_neighbor_picked_[ind + l] = 1;
          }
        }
      }

      for (int k = sp; k <= ep; k++)
      {
        if (cloud_label_[k] <= 0){
          surfaceCloudScan->push_back(extracted_cloud_->points[k]);
        }
      }
    }

    surfaceCloudScanDS->clear();
    down_size_filter_.setInputCloud(surfaceCloudScan);
    down_size_filter_.filter(*surfaceCloudScanDS);

    *surface_cloud_ += *surfaceCloudScanDS;
  }
}

void FeatureExtraction::free_cloud_info_memory()
{
  cloud_info_.start_ring_index.clear();
  cloud_info_.end_ring_index.clear();
  cloud_info_.point_col_index.clear();
  cloud_info_.point_range.clear();
}

void FeatureExtraction::publish_feature_cloud(
  const rclcpp::Time & now,
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerCloud,
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfaceCloud)
{
  // free cloud info memory
  free_cloud_info_memory();

  sensor_msgs::msg::PointCloud2 corner_cloud;
  pcl::toROSMsg(*corner_cloud_, corner_cloud);
  corner_cloud.header.stamp = now;
  corner_cloud.header.frame_id = lidar_frame_;
  pubCornerCloud->publish(corner_cloud);

  sensor_msgs::msg::PointCloud2 surface_cloud;
  pcl::toROSMsg(*surface_cloud_, surface_cloud);
  surface_cloud.header.stamp = now;
  surface_cloud.header.frame_id = lidar_frame_;
  pubSurfaceCloud->publish(surface_cloud);
}


} // namespace loam_feature_localization

