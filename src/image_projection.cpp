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

#include "loam_feature_localization/image_projection.hpp"

namespace loam_feature_localization
{

ImageProjection::ImageProjection(int N_SCAN, int Horizon_SCAN)
{
    n_scan_ = N_SCAN;
    horizon_scan_ = Horizon_SCAN;

    utils_ = std::make_shared<Utils>();

    allocate_memory();
    reset_parameters();

    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
}

void ImageProjection::allocate_memory()
{
  laser_cloud_in_.reset(new pcl::PointCloud<PointXYZIRT>());
  //    tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
  full_cloud_.reset(new pcl::PointCloud<PointType>());
  extracted_cloud_.reset(new pcl::PointCloud<PointType>());

  full_cloud_->points.resize(n_scan_ * horizon_scan_);

  cloud_info_.start_ring_index.assign(n_scan_, 0);
  cloud_info_.end_ring_index.assign(n_scan_, 0);

  cloud_info_.point_col_index.assign(n_scan_ * horizon_scan_, 0);
  cloud_info_.point_range.assign(n_scan_ * horizon_scan_, 0);

  reset_parameters();
}

void ImageProjection::reset_parameters()
{
  laser_cloud_in_->clear();
  extracted_cloud_->clear();
  // reset range matrix for range image projection
  range_mat_ = cv::Mat(n_scan_, horizon_scan_, CV_32F, cv::Scalar::all(FLT_MAX));

  imu_pointer_cur_ = 0;
  first_point_flag_ = true;
  odom_deskew_flag_ = false;

  for (int i = 0; i < queueLength; ++i) {
    imu_time_[i] = 0;
    imu_rot_x_[i] = 0;
    imu_rot_y_[i] = 0;
    imu_rot_z_[i] = 0;
  }
}

void ImageProjection::imu_handler(const sensor_msgs::msg::Imu::SharedPtr imuMsg)
{
  sensor_msgs::msg::Imu this_imu_ = utils_->imu_converter(*imuMsg);

  std::lock_guard<std::mutex> lock1(imu_lock_);
  imu_queue_.push_back(this_imu_);
}

void ImageProjection::odometry_handler(const nav_msgs::msg::Odometry::SharedPtr odometryMsg)
{
  std::lock_guard<std::mutex> lock2(odom_lock_);
  odom_queue_.push_back(*odometryMsg);
}

void ImageProjection::cloud_handler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg)
{
  if (!cache_point_cloud(laserCloudMsg)) return;

  if (!deskew_info()) return;

  project_point_cloud();

  cloud_extraction();

  publish_clouds();

  reset_parameters();
}

bool ImageProjection::cache_point_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr & laserCloudMsg)
{
  // cache point cloud
  cloud_queue_.push_back(*laserCloudMsg);
  if (cloud_queue_.size() <= 2) return false;

  //    for (sensor_msgs::PointCloud2ConstIterator<uint16_t> it(*laserCloudMsg, "channel");
  //         it != it.end(); ++it) {
  //      // TODO: do something with the values of x, y, z
  //      std::cout << it[0] << std::endl;
  //    }

  // convert cloud
  current_cloud_msg_ = std::move(cloud_queue_.front());
  cloud_queue_.pop_front();
//  if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX) {
//    pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
//  }
  //    else if (sensor == SensorType::OUSTER) {
  //      // Convert to Velodyne format
  //      pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
  //      laserCloudIn->points.resize(tmpOusterCloudIn->size());
  //      laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
  //      for (size_t i = 0; i < tmpOusterCloudIn->size(); i++) {
  //        auto & src = tmpOusterCloudIn->points[i];
  //        auto & dst = laserCloudIn->points[i];
  //        dst.x = src.x;
  //        dst.y = src.y;
  //        dst.z = src.z;
  //        dst.intensity = src.intensity;
  //        dst.channel = src.ring;
  //        dst.time = src.t * 1e-9f;
  //      }
  //    }
  else {
    RCLCPP_ERROR_STREAM(get_logger(), "Unknown sensor type: " << int(sensor));
    rclcpp::shutdown();
  }

  // get timestamp
  cloudHeader = currentCloudMsg.header;
  timeScanCur = stamp2Sec(cloudHeader.stamp);
  timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

  // remove Nan
  vector<int> indices;
  pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);

  // check dense flag
  if (laserCloudIn->is_dense == false) {
    RCLCPP_ERROR(
      get_logger(), "Point cloud is not in dense format, please remove NaN points first!");
    rclcpp::shutdown();
  }

  // check ring channel
  // we will skip the ring check in case of velodyne - as we calculate the ring value downstream
  // (line 572)
  if (ringFlag == 0) {
    ringFlag = -1;
    for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i) {
      if (currentCloudMsg.fields[i].name == "channel") {
        ringFlag = 1;
        break;
      }
    }
    if (ringFlag == -1) {
      if (sensor == SensorType::VELODYNE) {
        ringFlag = 2;
      } else {
        RCLCPP_ERROR(
          get_logger(),
          "Point cloud ring channel not available, please configure your point cloud data!");
        rclcpp::shutdown();
      }
    }
  }

  // check point time
  if (deskewFlag == 0) {
    deskewFlag = -1;
    for (auto & field : currentCloudMsg.fields) {
      if (field.name == "time_stamp" || field.name == "t") {
        deskewFlag = 1;
        break;
      }
    }
    if (deskewFlag == -1)
      RCLCPP_WARN(
        get_logger(),
        "Point cloud timestamp not available, deskew function disabled, system will drift "
        "significantly!");
  }

  return true;
}





} // namespace loam_feature_localization
