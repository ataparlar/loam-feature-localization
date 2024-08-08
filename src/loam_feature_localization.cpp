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

#include <pcl/filters/impl/filter.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <pcl/point_cloud.h>
#include <pcl/range_image/range_image.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>

namespace loam_feature_localization
{
LoamFeatureLocalization::LoamFeatureLocalization(const rclcpp::NodeOptions & options)
: Node("loam_feature_localization", options)
{
  this->declare_parameter("imu_topic", "");
  this->declare_parameter("odom_topic", "");
  this->declare_parameter("point_cloud_topic", "");
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
    std::bind(&LoamFeatureLocalization::imuHandler, this, std::placeholders::_1), imuOpt);
  subOdom = create_subscription<nav_msgs::msg::Odometry>(
    odom_topic_, rclcpp::SensorDataQoS(),
    std::bind(&LoamFeatureLocalization::odometryHandler, this, std::placeholders::_1), odomOpt);
  subLaserCloud = create_subscription<sensor_msgs::msg::PointCloud2>(
    point_cloud_topic_, rclcpp::SensorDataQoS(),
    std::bind(&LoamFeatureLocalization::cloudHandler, this, std::placeholders::_1), lidarOpt);

  pubRangeImage =
    create_publisher<sensor_msgs::msg::Image>("/range_image", rclcpp::SensorDataQoS());
  pubCloudBasic =
    create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_basic", rclcpp::SensorDataQoS());
  pubCloudUndistorted =
    create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_undistorted", rclcpp::SensorDataQoS());
  pubCornerCloud =
    create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_corner", rclcpp::SensorDataQoS());
  pubSurfaceCloud =
    create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_surface", rclcpp::SensorDataQoS());

  utils = std::make_shared<Utils>(lidar_imu_roll_, lidar_imu_pitch_, lidar_imu_yaw_,
                                  lidar_imu_x_, lidar_imu_y_, lidar_imu_z_);

  allocateMemory();
}

void LoamFeatureLocalization::imuHandler(const sensor_msgs::msg::Imu::SharedPtr imuMsg)
{
  sensor_msgs::msg::Imu thisImu = utils->imuConverter(*imuMsg);

  std::lock_guard<std::mutex> lock1(imuLock);
  imuQueue.push_back(thisImu);
}

void LoamFeatureLocalization::odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odometryMsg)
{
  std::lock_guard<std::mutex> lock2(odoLock);
  odomQueue.push_back(*odometryMsg);
}

void LoamFeatureLocalization::cloudHandler(
  const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg)
{
  sensor_msgs::msg::PointCloud2 point_cloud = *laserCloudMsg;
  point_cloud.header.stamp = this->get_clock()->now();
  point_cloud.header.frame_id = "lidar";
  pubCloudBasic->publish(point_cloud);

  // Image Projection

  if (!cachePointCloud(laserCloudMsg)) return;

  if (!deskewInfo()) return;

  projectPointCloud();
  cloudExtraction();
  publishImage();

  // Feature Extraction
  initializationValue();
  calculateSmoothness();
  markOccludedPoints();
  extractFeatures();

  // Feature Matching



  // -----------------
  publishClouds("map");
  resetParameters();

}




bool LoamFeatureLocalization::cachePointCloud(
  const sensor_msgs::msg::PointCloud2::SharedPtr & laserCloudMsg)
{
//  for (sensor_msgs::PointCloud2Iterator<uint16_t> it(*laserCloudMsg, "ring"); it != it.end(); ++it) {
//    it = ring_id;
//    if (counter % 2000) {
//      ring_id++;
//    }
//    counter++;
//  }

  // cache point cloud
  cloudQueue.push_back(*laserCloudMsg);
  if (cloudQueue.size() <= 2) return false;

  // convert cloud
  currentCloudMsg = std::move(cloudQueue.front());
  cloudQueue.pop_front();

  pcl::fromROSMsg(currentCloudMsg, *laserCloudIn);

//  uint16_t ring_id = 0;
//  int counter = 0;
//  for (auto & point : laserCloudIn->points) {
//    point.ring = ring_id;
//    if (counter % 2000 == 0) {
//      ring_id++;
//    }
//    counter++;
////    std::cout << point.ring << ", ";
//  }
//  //  std::cout << std::endl;


  //  if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX)
  //  {
  //    pcl::moveFromROSMsg(currentCloudMsg, **laserCloudIn);
  //  }
  //  else if (sensor == SensorType::OUSTER)
  //  {
  //    // Convert to Velodyne format
  //    pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
  //    *laserCloudIn->points.resize(tmpOusterCloudIn->size());
  //    *laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
  //    for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
  //    {
  //      auto &src = tmpOusterCloudIn->points[i];
  //      auto &dst = *laserCloudIn->points[i];
  //      dst.x = src.x;
  //      dst.y = src.y;
  //      dst.z = src.z;
  //      dst.intensity = src.intensity;
  //      dst.ring = src.ring;
  //      dst.time = src.t * 1e-9f;
  //    }
  //  }
  //  else
  //  {
  //    RCLCPP_ERROR_STREAM(get_logger(), "Unknown sensor type: " <<
  //    int(sensor)); rclcpp::shutdown();
  //  }

  // get timestamp
  cloudHeader = currentCloudMsg.header;
  timeScanCur = utils->stamp2Sec(cloudHeader.stamp);
  timeScanEnd = laserCloudIn->points.back().timestamp;

  // remove Nan
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);

  // check dense flag
  if (!laserCloudIn->is_dense) {
    RCLCPP_ERROR(
      get_logger(), "Point cloud is not in dense format, please remove NaN points first!");
    rclcpp::shutdown();
  }

  // check ring channel
  // we will skip the ring check in case of velodyne - as we calculate the ring
  // value downstream (line 572)
  if (ringFlag == 0) {
    ringFlag = -1;
    for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i) {
      if (currentCloudMsg.fields[i].name == "ring") {
        ringFlag = 1;
        break;
      }
    }
    if (ringFlag == -1) {
      ringFlag = 2;
      //      if (sensor == SensorType::VELODYNE) {
      //        ringFlag = 2;
      //      } else {
      //        RCLCPP_ERROR(get_logger(), "Point cloud ring channel not
      //        available, please configure your point cloud data!");
      //        rclcpp::shutdown();
      //      }
    }
  }

  // check point time
  if (deskewFlag == 0) {
    deskewFlag = -1;
    for (auto & field : currentCloudMsg.fields) {
      if (field.name == "timestamp" || field.name == "t") {
        deskewFlag = 1;
        break;
      }
    }
    if (deskewFlag == -1)
      RCLCPP_WARN(
        get_logger(),
        "Point cloud timestamp not available, deskew "
        "function disabled, system will drift "
        "significantly!");
  }

  return true;
}

bool LoamFeatureLocalization::deskewInfo()
{
  std::lock_guard<std::mutex> lock1(imuLock);
  std::lock_guard<std::mutex> lock2(odoLock);

  // make sure IMU data available for the scan
  if (
    imuQueue.empty() ||
    utils->stamp2Sec(imuQueue.front().header.stamp) > timeScanCur ||
    utils->stamp2Sec(imuQueue.back().header.stamp) < timeScanEnd)
  {
    RCLCPP_INFO(get_logger(), "Waiting for IMU data ...");
    return false;
  }

  imuDeskewInfo();

  //  odomDeskewInfo();

  return true;
}

void LoamFeatureLocalization::imuDeskewInfo()
{
  cloudInfo.imu_available = false;

  while (!imuQueue.empty()) {
    if (utils->stamp2Sec(imuQueue.front().header.stamp) < timeScanCur - 0.01)
      imuQueue.pop_front();
    else
      break;
  }

  if (imuQueue.empty()) return;

  imuPointerCur = 0;

  for (int i = 0; i < (int)imuQueue.size(); ++i) {
    sensor_msgs::msg::Imu thisImuMsg = imuQueue[i];
    double currentImuTime = utils->stamp2Sec(thisImuMsg.header.stamp);

    // get roll, pitch, and yaw estimation for this scan
    if (currentImuTime <= timeScanCur)
      utils->imuRPY2rosRPY(
        &thisImuMsg, &cloudInfo.imu_roll_init, &cloudInfo.imu_pitch_init, &cloudInfo.imu_yaw_init);
    if (currentImuTime > timeScanEnd + 0.01) break;

    if (imuPointerCur == 0) {
      imuRotX[0] = 0;
      imuRotY[0] = 0;
      imuRotZ[0] = 0;
      imuTime[0] = currentImuTime;
      ++imuPointerCur;
      continue;
    }

    // get angular velocity
    double angular_x, angular_y, angular_z;
    utils->imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

    // integrate rotation
    double timeDiff = currentImuTime - imuTime[imuPointerCur - 1];
    imuRotX[imuPointerCur] = imuRotX[imuPointerCur - 1] + angular_x * timeDiff;
    imuRotY[imuPointerCur] = imuRotY[imuPointerCur - 1] + angular_y * timeDiff;
    imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur - 1] + angular_z * timeDiff;
    imuTime[imuPointerCur] = currentImuTime;
    ++imuPointerCur;
  }

  --imuPointerCur;

  if (imuPointerCur <= 0) return;

  cloudInfo.imu_available = true;
}

void LoamFeatureLocalization::odomDeskewInfo()
{
  cloudInfo.odom_available = false;

  while (!odomQueue.empty()) {
    if (utils->stamp2Sec(odomQueue.front().header.stamp) < timeScanCur - 0.01)
      odomQueue.pop_front();
    else
      break;
  }

  if (odomQueue.empty()) return;

  if (utils->stamp2Sec(odomQueue.front().header.stamp) > timeScanCur) return;

  // get start odometry at the beinning of the scan
  nav_msgs::msg::Odometry startOdomMsg;

  for (int i = 0; i < (int)odomQueue.size(); ++i) {
    startOdomMsg = odomQueue[i];

    if (utils->stamp2Sec(startOdomMsg.header.stamp) < timeScanCur)
      continue;
    else
      break;
  }

  tf2::Quaternion orientation;
  tf2::fromMsg(startOdomMsg.pose.pose.orientation, orientation);

  double roll, pitch, yaw;
  tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

  // Initial guess used in mapOptimization
  cloudInfo.initial_guess_x = startOdomMsg.pose.pose.position.x;
  cloudInfo.initial_guess_y = startOdomMsg.pose.pose.position.y;
  cloudInfo.initial_guess_z = startOdomMsg.pose.pose.position.z;
  cloudInfo.initial_guess_roll = roll;
  cloudInfo.initial_guess_pitch = pitch;
  cloudInfo.initial_guess_yaw = yaw;

  cloudInfo.odom_available = true;

  // get end odometry at the end of the scan
  odomDeskewFlag = false;

  if (utils->stamp2Sec(odomQueue.back().header.stamp) < timeScanEnd) return;

  nav_msgs::msg::Odometry endOdomMsg;

  for (int i = 0; i < (int)odomQueue.size(); ++i) {
    endOdomMsg = odomQueue[i];

    if (utils->stamp2Sec(endOdomMsg.header.stamp) < timeScanEnd)
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
    transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

  odomDeskewFlag = true;
}

void LoamFeatureLocalization::projectPointCloud()
{
  HSV = cv::Mat(N_SCAN_, Horizon_SCAN_, CV_8UC3, cv::Scalar::all(FLT_MAX));

  int cloudSize = laserCloudIn->points.size();
  // range image projection
  for (int i = 0; i < cloudSize; ++i) {
    PointType thisPoint;
    thisPoint.x = laserCloudIn->points[i].x;
    thisPoint.y = laserCloudIn->points[i].y;
    thisPoint.z = laserCloudIn->points[i].z;
    thisPoint.intensity = laserCloudIn->points[i].intensity;

    float range = utils->pointDistance(thisPoint);
    if (range < lidar_min_range_ || range > lidar_max_range_) continue;

    int rowIdn = laserCloudIn->points[i].ring;
    // if sensor is a velodyne (ringFlag = 2) calculate rowIdn based on number of scans
    if (ringFlag == 2) {
      float verticalAngle =
        atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 /
        M_PI;
      rowIdn = (verticalAngle + (N_SCAN_ - 1)) / 2.0;
    }

    if (rowIdn < 0 || rowIdn >= N_SCAN_) continue;

    //    if (rowIdn % downsampleRate != 0)
    //      continue;

    int columnIdn = -1;
    float horizonAngle;
    static float ang_res_x;

    horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
    ang_res_x = 360.0 / float(Horizon_SCAN_);
    columnIdn = -round((horizonAngle) / ang_res_x) + Horizon_SCAN_ / 2;
    //    columnIdn = round((horizonAngle)/ang_res_x);
    if (columnIdn >= Horizon_SCAN_) columnIdn -= Horizon_SCAN_;

    if (columnIdn < 0 || columnIdn >= Horizon_SCAN_) continue;

    if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX) continue;

    thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].timestamp);
    cloudDeskewed->push_back(thisPoint);

    rangeMat.at<float>(rowIdn, columnIdn) = range;
    //                rangeMat.at<float>(rowIdn, columnIdn) = horizonAngle;

    //            uchar hue = static_cast<uchar>((horizonAngle * 180.0) / 360 + 90);
    uchar hue = static_cast<uchar>((range * 180.0) / 60);
    HSV.at<cv::Vec3b>(rowIdn, columnIdn) = cv::Vec3b(hue, 255.0, 255.0);

    int index = columnIdn + rowIdn * Horizon_SCAN_;
    fullCloud->points[index] = thisPoint;
  }
}

PointType LoamFeatureLocalization::deskewPoint(PointType * point, double relTime)
{
  if (deskewFlag == -1 || cloudInfo.imu_available == false) return *point;

  double pointTime = timeScanCur + relTime;

  float rotXCur, rotYCur, rotZCur;
  findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

  float posXCur, posYCur, posZCur;
  findPosition(relTime, &posXCur, &posYCur, &posZCur);

  if (firstPointFlag == true) {
    transStartInverse =
      (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
    firstPointFlag = false;
  }

  // transform points to start
  Eigen::Affine3f transFinal =
    pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
  Eigen::Affine3f transBt = transStartInverse * transFinal;

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

void LoamFeatureLocalization::findRotation(
  double pointTime, float * rotXCur, float * rotYCur, float * rotZCur)
{
  *rotXCur = 0;
  *rotYCur = 0;
  *rotZCur = 0;

  int imuPointerFront = 0;
  while (imuPointerFront < imuPointerCur) {
    if (pointTime < imuTime[imuPointerFront]) break;
    ++imuPointerFront;
  }

  if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0) {
    *rotXCur = imuRotX[imuPointerFront];
    *rotYCur = imuRotY[imuPointerFront];
    *rotZCur = imuRotZ[imuPointerFront];
  } else {
    int imuPointerBack = imuPointerFront - 1;
    double ratioFront =
      (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
    double ratioBack =
      (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
    *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
    *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
    *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
  }
}

void LoamFeatureLocalization::findPosition(
  double relTime, float * posXCur, float * posYCur, float * posZCur)
{
  *posXCur = 0;
  *posYCur = 0;
  *posZCur = 0;

  if (cloudInfo.odom_available == false || odomDeskewFlag == false) return;

  float ratio = relTime / (timeScanEnd - timeScanCur);

  *posXCur = ratio * odomIncreX;
  *posYCur = ratio * odomIncreY;
  *posZCur = ratio * odomIncreZ;
}

void LoamFeatureLocalization::cloudExtraction()
{
  int count = 0;
  // extract segmented cloud for lidar odometry
  for (int i = 0; i < N_SCAN_; ++i) {
    cloudInfo.start_ring_index[i] = count - 1 + 5;
    for (int j = 0; j < Horizon_SCAN_; ++j) {
      if (rangeMat.at<float>(i, j) != FLT_MAX) {
        // mark the points' column index for marking occlusion later
        cloudInfo.point_col_index[count] = j;
        // save range info
        cloudInfo.point_range[count] = rangeMat.at<float>(i, j);
        // save extracted cloud
        extractedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN_]);
        // size of extracted cloud
        ++count;
      }
    }
    cloudInfo.end_ring_index[i] = count - 1 - 5;
  }
}

void LoamFeatureLocalization::allocateMemory()
{

  laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
  tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
  fullCloud.reset(new pcl::PointCloud<PointType>());
  extractedCloud.reset(new pcl::PointCloud<PointType>());
  cloudDeskewed.reset(new pcl::PointCloud<PointType>());

  fullCloud->resize(N_SCAN_ * Horizon_SCAN_);

  cloudInfo.start_ring_index.assign(N_SCAN_, 0);
  cloudInfo.end_ring_index.assign(N_SCAN_, 0);

  cloudInfo.point_col_index.assign(N_SCAN_ * Horizon_SCAN_, 0);
  cloudInfo.point_range.assign(N_SCAN_ * Horizon_SCAN_, 0);

  resetParameters();
}

void LoamFeatureLocalization::resetParameters()
{
  laserCloudIn->clear();
  extractedCloud->clear();
  cloudDeskewed->clear();
  // reset range matrix for range image projection
  rangeMat = cv::Mat(N_SCAN_, Horizon_SCAN_, CV_32F, FLT_MAX);

  imuPointerCur = 0;
  firstPointFlag = true;
  odomDeskewFlag = false;

  for (int i = 0; i < queueLength; ++i) {
    imuTime[i] = 0;
    imuRotX[i] = 0;
    imuRotY[i] = 0;
    imuRotZ[i] = 0;
  }
}

void LoamFeatureLocalization::publishClouds(std::string frame_name)
{
  sensor_msgs::msg::PointCloud2 cloud_undistorted;
  pcl::toROSMsg(*cloudDeskewed, cloud_undistorted);
  cloud_undistorted.header.stamp = this->get_clock()->now();
  cloud_undistorted.header.frame_id = frame_name;
  pubCloudUndistorted->publish(cloud_undistorted);

  sensor_msgs::msg::PointCloud2 cloud_corner;
  pcl::toROSMsg(*cornerCloud, cloud_corner);
  cloud_corner.header.stamp = this->get_clock()->now();
  cloud_corner.header.frame_id = frame_name;
  pubCornerCloud->publish(cloud_corner);

  sensor_msgs::msg::PointCloud2 cloud_surface;
  pcl::toROSMsg(*surfaceCloud, cloud_surface);
  cloud_surface.header.stamp = this->get_clock()->now();
  cloud_surface.header.frame_id = frame_name;
  pubSurfaceCloud->publish(cloud_surface);
}

void LoamFeatureLocalization::publishImage()
{
  cv::Mat BGR;
  cv::cvtColor(HSV, BGR, cv::COLOR_HSV2BGR);

  cv::Mat bgr_resized;
  cv::resize(BGR, bgr_resized, cv::Size(), 1.0, 20.0);

  cv_bridge::CvImage cv_image;
  cv_image.header.frame_id = "map";
  cv_image.header.stamp = this->get_clock()->now();
  cv_image.encoding = "bgr8";
  cv_image.image = bgr_resized;

  sensor_msgs::msg::Image image;
  cv_image.toImageMsg(image);

  pubRangeImage->publish(image);
}


// Feature Extraction

void LoamFeatureLocalization::initializationValue()
{
  cloudSmoothness.resize(N_SCAN_*Horizon_SCAN_);

  downSizeFilter.setLeafSize(odometry_surface_leaf_size_, odometry_surface_leaf_size_, odometry_surface_leaf_size_);

//  extractedCloud.reset(new pcl::PointCloud<PointType>());
  cornerCloud.reset(new pcl::PointCloud<PointType>());
  surfaceCloud.reset(new pcl::PointCloud<PointType>());

  cloudCurvature = new float[N_SCAN_*Horizon_SCAN_];
  cloudNeighborPicked = new int[N_SCAN_*Horizon_SCAN_];
  cloudLabel = new int[N_SCAN_*Horizon_SCAN_];
}

void LoamFeatureLocalization::calculateSmoothness()
{
  int cloudSize = extractedCloud->points.size();
  for (int i = 5; i < cloudSize - 5; i++)
  {
    float diffRange = cloudInfo.point_range[i-5] + cloudInfo.point_range[i-4]
                      + cloudInfo.point_range[i-3] + cloudInfo.point_range[i-2]
                      + cloudInfo.point_range[i-1] - cloudInfo.point_range[i] * 10
                      + cloudInfo.point_range[i+1] + cloudInfo.point_range[i+2]
                      + cloudInfo.point_range[i+3] + cloudInfo.point_range[i+4]
                      + cloudInfo.point_range[i+5];

    cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;

    cloudNeighborPicked[i] = 0;
    cloudLabel[i] = 0;
    // cloudSmoothness for sorting
    cloudSmoothness[i].value = cloudCurvature[i];
    cloudSmoothness[i].ind = i;
  }
}

void LoamFeatureLocalization::markOccludedPoints()
{
  int cloudSize = extractedCloud->points.size();
  // mark occluded points and parallel beam points
  for (int i = 5; i < cloudSize - 6; ++i)
  {
    // occluded points
    float depth1 = cloudInfo.point_range[i];
    float depth2 = cloudInfo.point_range[i+1];
    int columnDiff = std::abs(int(cloudInfo.point_col_index[i+1] - cloudInfo.point_col_index[i]));
    if (columnDiff < 10){
      // 10 pixel diff in range image
      if (depth1 - depth2 > 0.3){
        cloudNeighborPicked[i - 5] = 1;
        cloudNeighborPicked[i - 4] = 1;
        cloudNeighborPicked[i - 3] = 1;
        cloudNeighborPicked[i - 2] = 1;
        cloudNeighborPicked[i - 1] = 1;
        cloudNeighborPicked[i] = 1;
      }else if (depth2 - depth1 > 0.3){
        cloudNeighborPicked[i + 1] = 1;
        cloudNeighborPicked[i + 2] = 1;
        cloudNeighborPicked[i + 3] = 1;
        cloudNeighborPicked[i + 4] = 1;
        cloudNeighborPicked[i + 5] = 1;
        cloudNeighborPicked[i + 6] = 1;
      }
    }
    // parallel beam
    float diff1 = std::abs(float(cloudInfo.point_range[i-1] - cloudInfo.point_range[i]));
    float diff2 = std::abs(float(cloudInfo.point_range[i+1] - cloudInfo.point_range[i]));

    if (diff1 > 0.02 * cloudInfo.point_range[i] && diff2 > 0.02 * cloudInfo.point_range[i])
      cloudNeighborPicked[i] = 1;
  }
}

void LoamFeatureLocalization::extractFeatures()
{
  cornerCloud->clear();
  surfaceCloud->clear();

  pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

  for (int i = 0; i < N_SCAN_; i++)
  {
    surfaceCloudScan->clear();

    for (int j = 0; j < 6; j++)
    {

      int sp = (cloudInfo.start_ring_index[i] * (6 - j) + cloudInfo.end_ring_index[i] * j) / 6;
      int ep = (cloudInfo.start_ring_index[i] * (5 - j) + cloudInfo.end_ring_index[i] * (j + 1)) / 6 - 1;

      if (sp >= ep)
        continue;

      std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--)
      {
        int ind = cloudSmoothness[k].ind;
        if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edge_threshold_)
        {
          largestPickedNum++;
          if (largestPickedNum <= 20){
            cloudLabel[ind] = 1;
            cornerCloud->push_back(extractedCloud->points[ind]);
          } else {
            break;
          }

          cloudNeighborPicked[ind] = 1;
          for (int l = 1; l <= 5; l++)
          {
            int columnDiff = std::abs(int(cloudInfo.point_col_index[ind + l] - cloudInfo.point_col_index[ind + l - 1]));
            if (columnDiff > 10)
              break;
            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--)
          {
            int columnDiff = std::abs(int(cloudInfo.point_col_index[ind + l] - cloudInfo.point_col_index[ind + l + 1]));
            if (columnDiff > 10)
              break;
            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      for (int k = sp; k <= ep; k++)
      {
        int ind = cloudSmoothness[k].ind;
        if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surface_threshold_)
        {

          cloudLabel[ind] = -1;
          cloudNeighborPicked[ind] = 1;

          for (int l = 1; l <= 5; l++) {
            int columnDiff = std::abs(int(cloudInfo.point_col_index[ind + l] - cloudInfo.point_col_index[ind + l - 1]));
            if (columnDiff > 10)
              break;

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            int columnDiff = std::abs(int(cloudInfo.point_col_index[ind + l] - cloudInfo.point_col_index[ind + l + 1]));
            if (columnDiff > 10)
              break;

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      for (int k = sp; k <= ep; k++)
      {
        if (cloudLabel[k] <= 0){
          surfaceCloudScan->push_back(extractedCloud->points[k]);
        }
      }
    }

    surfaceCloudScanDS->clear();
    downSizeFilter.setInputCloud(surfaceCloudScan);
    downSizeFilter.filter(*surfaceCloudScanDS);

    *surfaceCloud += *surfaceCloudScanDS;
  }
}

void LoamFeatureLocalization::freeCloudInfoMemory()
{
  cloudInfo.start_ring_index.clear();
  cloudInfo.end_ring_index.clear();
  cloudInfo.point_col_index.clear();
  cloudInfo.point_range.clear();
}

//void LoamFeatureLocalization::publishFeatureCloud()
//{
//  // free cloud info memory
//  freeCloudInfoMemory();
//  // save newly extracted features
//  cloudInfo.cloud_corner = publishCloud(pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
//  cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
//  // publish to mapOptimization
//  pubLaserCloudInfo->publish(cloudInfo);
//}





}  // namespace loam_feature_localization

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(loam_feature_localization::LoamFeatureLocalization)
