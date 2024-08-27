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

#include "pcl_conversions/pcl_conversions.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"

#include <pcl/common/impl/eigen.hpp>
#include <pcl/filters/filter.h>

#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace loam_feature_localization
{

ImageProjection::ImageProjection(  const Utils::SharedPtr & utils,
  int N_SCAN, int Horizon_SCAN, double lidar_max_range, double lidar_min_range,
  std::string lidar_frame)
{
  n_scan_ = N_SCAN;
  horizon_scan_ = Horizon_SCAN;
  lidar_max_range_ = lidar_max_range;
  lidar_min_range_ = lidar_min_range;
  lidar_frame_ = lidar_frame;

//  utils_ = std::make_shared<Utils>();
  utils_ = utils;

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
  extracted_cloud_to_pub_.reset(new pcl::PointCloud<PointType>());

  full_cloud_->points.resize(n_scan_ * horizon_scan_);

  cloud_info.start_ring_index.assign(n_scan_, 0);
  cloud_info.end_ring_index.assign(n_scan_, 0);

  cloud_info.point_col_index.assign(n_scan_ * horizon_scan_, 0);
  cloud_info.point_range.assign(n_scan_ * horizon_scan_, 0);

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
  sensor_msgs::msg::Imu this_imu = utils_->imuConverter(*imuMsg);

//  std::lock_guard<std::mutex> lock1(imu_lock_);
  imu_queue_.push_back(this_imu);
}

void ImageProjection::odometry_handler(const nav_msgs::msg::Odometry::SharedPtr odometryMsg)
{
//  std::lock_guard<std::mutex> lock2(odom_lock_);
  odom_queue_.push_back(*odometryMsg);
}

void ImageProjection::cloud_handler(
  const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg, rclcpp::Logger logger_,
  rclcpp::Time now, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_deskewed)
{
  if (!cache_point_cloud(laserCloudMsg, logger_)) return;

  if (!deskew_info(logger_)) return;

  project_point_cloud();

  cloud_extraction();

  publish_clouds(now, pub_cloud_deskewed);

  range_mat_for_vis_ = prepare_visualization_image(range_mat_);

  extracted_cloud_to_pub_ = extracted_cloud_;

  reset_parameters();
}

bool ImageProjection::cache_point_cloud(
  const sensor_msgs::msg::PointCloud2::SharedPtr & laserCloudMsg, rclcpp::Logger logger_)
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
  //  else {
  //    RCLCPP_ERROR_STREAM(get_logger(), "Unknown sensor type: " << int(sensor));
  //    rclcpp::shutdown();
  //  }

  pcl::moveFromROSMsg(current_cloud_msg_, *laser_cloud_in_);

  // get timestamp
  cloud_header_ = current_cloud_msg_.header;
  time_scan_cur_ = utils_->stamp2Sec(cloud_header_.stamp);
  time_scan_end_ = time_scan_cur_ + laser_cloud_in_->points.back().time;

  // remove Nan
//  std::vector<int> indices;
//  pcl::removeNaNFromPointCloud(*laser_cloud_in_, *laser_cloud_in_, indices);

  // check dense flag
  if (laser_cloud_in_->is_dense == false) {
    RCLCPP_ERROR(logger_, "Point cloud is not in dense format, please remove NaN points first!");
    rclcpp::shutdown();
  }

  // check ring channel
  // we will skip the ring check in case of velodyne - as we calculate the ring value downstream
  // (line 572)
  if (ring_flag_ == 0) {
    ring_flag_ = -1;
    for (int i = 0; i < (int)current_cloud_msg_.fields.size(); ++i) {
      if (current_cloud_msg_.fields[i].name == "channel") {
        ring_flag_ = 1;
        break;
      }
    }
    //    if (ring_flag_ == -1) {
    //      if (sensor == SensorType::VELODYNE) {
    //        ringFlag = 2;
    //      } else {
    //        RCLCPP_ERROR(
    //          get_logger(),
    //          "Point cloud ring channel not available, please configure your point cloud data!");
    //        rclcpp::shutdown();
    //      }
    //    }
  }

  // check point time
  if (deskew_flag_ == 0) {
    deskew_flag_ = -1;
    for (auto & field : current_cloud_msg_.fields) {
      if (field.name == "time_stamp" || field.name == "t") {
        deskew_flag_ = 1;
        break;
      }
    }
    if (deskew_flag_ == -1)
      RCLCPP_WARN(
        logger_,
        "Point cloud timestamp not available, deskew function disabled, system will drift "
        "significantly!");
  }

  return true;
}

bool ImageProjection::deskew_info(rclcpp::Logger logger_)
{
//  std::lock_guard<std::mutex> lock1(imu_lock_);
//  std::lock_guard<std::mutex> lock2(odom_lock_);

  // make sure IMU data available for the scan
  if (
    imu_queue_.empty() || utils_->stamp2Sec(imu_queue_.front().header.stamp) > time_scan_cur_ ||
    utils_->stamp2Sec(imu_queue_.back().header.stamp) < time_scan_end_) {
    RCLCPP_INFO(logger_, "Waiting for IMU data ...");
    return false;
  }

  imu_deskew_info();

  odom_deskew_info();

  return true;
}

void ImageProjection::imu_deskew_info()
{
  cloud_info.imu_available = false;

  while (!imu_queue_.empty()) {
    if (utils_->stamp2Sec(imu_queue_.front().header.stamp) < time_scan_cur_ - 0.01)
      imu_queue_.pop_front();
    else
      break;
  }

  if (imu_queue_.empty()) return;

  imu_pointer_cur_ = 0;

  for (int i = 0; i < (int)imu_queue_.size(); ++i) {
    sensor_msgs::msg::Imu thisImuMsg = imu_queue_[i];
    double currentimu_time_ = utils_->stamp2Sec(thisImuMsg.header.stamp);

    // get roll, pitch, and yaw estimation for this scan
    if (currentimu_time_ <= time_scan_cur_)
      utils_->imuRPY2rosRPY(
        &thisImuMsg, &cloud_info.imu_roll_init, &cloud_info.imu_pitch_init,
        &cloud_info.imu_yaw_init);
    if (currentimu_time_ > time_scan_end_ + 0.01) break;

    if (imu_pointer_cur_ == 0) {
      imu_rot_x_[0] = 0;
      imu_rot_y_[0] = 0;
      imu_rot_z_[0] = 0;
      imu_time_[0] = currentimu_time_;
      ++imu_pointer_cur_;
      continue;
    }

    // get angular velocity
    double angular_x, angular_y, angular_z;
    utils_->imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

    // integrate rotation
    double timeDiff = currentimu_time_ - imu_time_[imu_pointer_cur_ - 1];
    imu_rot_x_[imu_pointer_cur_] = imu_rot_x_[imu_pointer_cur_ - 1] + angular_x * timeDiff;
    imu_rot_y_[imu_pointer_cur_] = imu_rot_y_[imu_pointer_cur_ - 1] + angular_y * timeDiff;
    imu_rot_z_[imu_pointer_cur_] = imu_rot_z_[imu_pointer_cur_ - 1] + angular_z * timeDiff;
    imu_time_[imu_pointer_cur_] = currentimu_time_;
    ++imu_pointer_cur_;
  }

  --imu_pointer_cur_;

  if (imu_pointer_cur_ <= 0) return;

  cloud_info.imu_available = true;
}

void ImageProjection::odom_deskew_info()
{
  cloud_info.odom_available = false;

  while (!odom_queue_.empty()) {
    if (utils_->stamp2Sec(odom_queue_.front().header.stamp) < time_scan_cur_ - 0.01)
      odom_queue_.pop_front();
    else
      break;
  }

  if (odom_queue_.empty()) return;

  if (utils_->stamp2Sec(odom_queue_.front().header.stamp) > time_scan_cur_) return;

  // get start odometry at the beinning of the scan
  nav_msgs::msg::Odometry startOdomMsg;

  for (int i = 0; i < (int)odom_queue_.size(); ++i) {
    startOdomMsg = odom_queue_[i];

    if (utils_->stamp2Sec(startOdomMsg.header.stamp) < time_scan_cur_)
      continue;
    else
      break;
  }

  tf2::Quaternion orientation;
  tf2::fromMsg(startOdomMsg.pose.pose.orientation, orientation);

  double roll, pitch, yaw;
  tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

  // Initial guess used in mapOptimization
  cloud_info.initial_guess_x = startOdomMsg.pose.pose.position.x;
  cloud_info.initial_guess_y = startOdomMsg.pose.pose.position.y;
  cloud_info.initial_guess_z = startOdomMsg.pose.pose.position.z;
  cloud_info.initial_guess_roll = roll;
  cloud_info.initial_guess_pitch = pitch;
  cloud_info.initial_guess_yaw = yaw;

  cloud_info.odom_available = true;

  // get end odometry at the end of the scan
  odom_deskew_flag_ = false;

  if (utils_->stamp2Sec(odom_queue_.back().header.stamp) < time_scan_end_) return;

  nav_msgs::msg::Odometry endOdomMsg;

  for (int i = 0; i < (int)odom_queue_.size(); ++i) {
    endOdomMsg = odom_queue_[i];

    if (utils_->stamp2Sec(endOdomMsg.header.stamp) < time_scan_end_)
      continue;
    else
      break;
  }

  if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
    return;

  Eigen::Affine3f transBegin = pcl::getTransformation(
    startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y,
    startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

  tf2::fromMsg(endOdomMsg.pose.pose.orientation, orientation);
  tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
  Eigen::Affine3f transEnd = pcl::getTransformation(
    endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y,
    endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

  Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

  float rollIncre, pitchIncre, yawIncre;
  pcl::getTranslationAndEulerAngles(
    transBt, odom_incre_x_, odom_incre_y_, odom_incre_z_, rollIncre, pitchIncre, yawIncre);

  odom_deskew_flag_ = true;
}

void ImageProjection::find_rotation(
  double pointTime, float * rot_x_cur, float * rot_y_cur, float * rot_z_cur)
{
  *rot_x_cur = 0;
  *rot_y_cur = 0;
  *rot_z_cur = 0;

  int imuPointerFront = 0;
  while (imuPointerFront < imu_pointer_cur_) {
    if (pointTime < imu_time_[imuPointerFront]) break;
    ++imuPointerFront;
  }

  if (pointTime > imu_time_[imuPointerFront] || imuPointerFront == 0) {
    *rot_x_cur = imu_rot_x_[imuPointerFront];
    *rot_y_cur = imu_rot_y_[imuPointerFront];
    *rot_z_cur = imu_rot_z_[imuPointerFront];
  } else {
    int imuPointerBack = imuPointerFront - 1;
    double ratioFront = (pointTime - imu_time_[imuPointerBack]) /
                        (imu_time_[imuPointerFront] - imu_time_[imuPointerBack]);
    double ratioBack = (imu_time_[imuPointerFront] - pointTime) /
                       (imu_time_[imuPointerFront] - imu_time_[imuPointerBack]);
    *rot_x_cur = imu_rot_x_[imuPointerFront] * ratioFront + imu_rot_x_[imuPointerBack] * ratioBack;
    *rot_y_cur = imu_rot_y_[imuPointerFront] * ratioFront + imu_rot_y_[imuPointerBack] * ratioBack;
    *rot_z_cur = imu_rot_z_[imuPointerFront] * ratioFront + imu_rot_z_[imuPointerBack] * ratioBack;
  }
}

void ImageProjection::find_position(
  double relTime, float * pos_x_cur_, float * pos_y_cur_, float * pos_z_cur_)
{
  *pos_x_cur_ = 0;
  *pos_y_cur_ = 0;
  *pos_z_cur_ = 0;

  // If the sensor moves relatively slow, like walking speed, positional deskew seems to have
  // little benefits. Thus code below is commented.

  if (cloud_info.odom_available == false || odom_deskew_flag_ == false) return;

  float ratio = relTime / (time_scan_end_ - time_scan_cur_);

  *pos_x_cur_ = ratio * odom_incre_x_;
  *pos_y_cur_ = ratio * odom_incre_y_;
  *pos_z_cur_ = ratio * odom_incre_z_;
}

PointType ImageProjection::deskew_point(PointType * point, double relTime)
{
  if (deskew_flag_ == -1 || cloud_info.imu_available == false) return *point;

  double pointTime = time_scan_cur_ + relTime;

  float rotXCur, rotYCur, rotZCur;
  find_rotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

  float posXCur, posYCur, posZCur;
  find_rotation(relTime, &posXCur, &posYCur, &posZCur);

  if (first_point_flag_ == true) {
    trans_start_inverse_ =
      (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
    first_point_flag_ = false;
  }

  // transform points to start
  Eigen::Affine3f transFinal =
    pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
  Eigen::Affine3f transBt = trans_start_inverse_ * transFinal;

  PointType newPoint;
  newPoint.x =
    transBt(0, 0) * point->x + transBt(0, 1) * point->y + transBt(0, 2) * point->z + transBt(0, 3);
  newPoint.y =
    transBt(1, 0) * point->x + transBt(1, 1) * point->y + transBt(1, 2) * point->z + transBt(1, 3);
  newPoint.z =
    transBt(2, 0) * point->x + transBt(2, 1) * point->y + transBt(2, 2) * point->z + transBt(2, 3);
  newPoint.intensity = point->intensity;

  return newPoint;
}

void ImageProjection::project_point_cloud()
{
  int cloudSize = laser_cloud_in_->points.size();
  // range image projection
  for (int i = 0; i < cloudSize; ++i) {
    PointType thisPoint;
    thisPoint.x = laser_cloud_in_->points[i].x;
    thisPoint.y = laser_cloud_in_->points[i].y;
    thisPoint.z = laser_cloud_in_->points[i].z;
    thisPoint.intensity = laser_cloud_in_->points[i].intensity;

    float range = utils_->pointDistance(thisPoint);
    if (range < lidar_min_range_ || range > lidar_max_range_) continue;

    int rowIdn = laser_cloud_in_->points[i].channel;
    // if sensor is a velodyne (ringFlag = 2) calculate rowIdn based on number of scans
    if (ring_flag_ == 2) {
      float verticalAngle =
        atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 /
        M_PI;
      rowIdn = (verticalAngle + (n_scan_ - 1)) / 2.0;
    }

    if (rowIdn < 0 || rowIdn >= n_scan_) continue;

    //    if (rowIdn % downsampleRate != 0) continue;

    int columnIdn = -1;
    //    if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER) {
    float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
    static float ang_res_x = 360.0 / float(horizon_scan_);
    columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + horizon_scan_ / 2;
    if (columnIdn >= horizon_scan_) columnIdn -= horizon_scan_;
    //    } else if (sensor == SensorType::LIVOX) {
    //      columnIdn = columnIdnCountVec[rowIdn];
    //      columnIdnCountVec[rowIdn] += 1;
    //    }

    if (columnIdn < 0 || columnIdn >= horizon_scan_) continue;

    if (range_mat_.at<float>(rowIdn, columnIdn) != FLT_MAX) continue;

    thisPoint = deskew_point(&thisPoint, laser_cloud_in_->points[i].time);

    range_mat_.at<float>(rowIdn, columnIdn) = range;

    int index = columnIdn + rowIdn * horizon_scan_;
    full_cloud_->points[index] = thisPoint;
  }
}

void ImageProjection::cloud_extraction()
{
  int count = 0;
  // extract segmented cloud for lidar odometry
  for (int i = 0; i < n_scan_; ++i) {
    cloud_info.start_ring_index[i] = count - 1 + 5;
    for (int j = 0; j < horizon_scan_; ++j) {
      if (range_mat_.at<float>(i, j) != FLT_MAX) {
        // mark the points' column index for marking occlusion later
        cloud_info.point_col_index[count] = j;
        // save range info
        cloud_info.point_range[count] = range_mat_.at<float>(i, j);
        // save extracted cloud
        extracted_cloud_->push_back(full_cloud_->points[j + i * horizon_scan_]);
        // size of extracted cloud
        ++count;
      }
    }
    cloud_info.end_ring_index[i] = count - 1 - 5;
  }
}

void ImageProjection::publish_clouds(
  const rclcpp::Time & now,
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_deskewed)
{
  //  cloud_info.header = cloudHeader;
  //  cloud_info.cloud_deskewed =
  //    publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
  //  pubLaserCloudInfo->publish(cloudInfo);

  sensor_msgs::msg::PointCloud2 cloud_deskewed;
  pcl::toROSMsg(*full_cloud_, cloud_deskewed);
  cloud_deskewed.header.stamp = now;
  cloud_deskewed.header.frame_id = lidar_frame_;
  pub_cloud_deskewed->publish(cloud_deskewed);
}

sensor_msgs::msg::Image ImageProjection::prepare_visualization_image(const cv::Mat & rangeMat) {
  cv::Mat normalized_image(rangeMat.rows, rangeMat.cols, CV_8UC1, 0.0);
  for (int col = 0; col < rangeMat.cols; ++col) {
    for (int row = 0; row < rangeMat.rows; ++row) {
      normalized_image.at<uchar>(row, col) =
        static_cast<uchar>((rangeMat.at<float>(row, col) * 180.0) / 60);
    }
  }

  cv::Mat hsv_image(normalized_image.size(), CV_8UC3, cv::Vec3b(0.0, 255.0, 255.0));
  for (int col = 0; col < normalized_image.cols; ++col) {
    for (int row = 0; row < normalized_image.rows; ++row) {
      uchar hue = normalized_image.at<uchar>(row, col);
      if (hue == 0) {
        hsv_image.at<cv::Vec3b>(row, col) = cv::Vec3b(hue, 0, 0);  // Full saturation and value
      } else {
        hsv_image.at<cv::Vec3b>(row, col) = cv::Vec3b(hue, 255, 255);  // Full saturation and value
      }
    }
  }

  cv::Mat BGR;
  cv::cvtColor(hsv_image, BGR, cv::COLOR_HSV2BGR);

  cv::Mat bgr_resized;
  cv::resize(BGR, bgr_resized, cv::Size(), 1.0, 20.0);

  cv_bridge::CvImage cv_image;
  cv_image.header.frame_id = "map";
//  cv_image.header.stamp = this->get_clock()->now();
  cv_image.encoding = "bgr8";
  cv_image.image = bgr_resized;

  sensor_msgs::msg::Image image;
  cv_image.toImageMsg(image);

  return image;
}

}  // namespace loam_feature_localization
