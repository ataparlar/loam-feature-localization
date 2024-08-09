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
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

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
    std::bind(&LoamFeatureLocalization::imuHandler, this, std::placeholders::_1), imuOpt);
  subOdom = create_subscription<nav_msgs::msg::Odometry>(
    odom_topic_, rclcpp::SensorDataQoS(),
    std::bind(&LoamFeatureLocalization::odometryHandler, this, std::placeholders::_1), odomOpt);
  subLaserCloud = create_subscription<sensor_msgs::msg::PointCloud2>(
    point_cloud_topic_, rclcpp::SensorDataQoS(),
    std::bind(&LoamFeatureLocalization::cloudHandler, this, std::placeholders::_1), lidarOpt);

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
  pubLaserOdometryGlobal =
    create_publisher<nav_msgs::msg::Odometry>("/laser_odom", rclcpp::SensorDataQoS());

  br = std::make_unique<tf2_ros::TransformBroadcaster>(this);

  utils = std::make_shared<Utils>(
    lidar_imu_roll_, lidar_imu_pitch_, lidar_imu_yaw_, lidar_imu_x_, lidar_imu_y_, lidar_imu_z_);

  gtsam::ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.1;
  parameters.relinearizeSkip = 1;
  isam = new gtsam::ISAM2(parameters);

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

//  static double timeLastProcessing = -1;
//  if (timeLaserInfoCur - timeLastProcessing >= 1) {
//    timeLastProcessing = timeLaserInfoCur;

    updateInitialGuess();

    extractSurroundingKeyFrames();

    downsampleCurrentScan();

    scan2MapOptimization();

    saveKeyFramesAndFactor();

    correctPoses();

    publishOdometry();

    //    publishFrames();
//  }

  // -----------------
  publishClouds(output_odometry_frame_);
  resetParameters();
}

bool LoamFeatureLocalization::cachePointCloud(
  const sensor_msgs::msg::PointCloud2::SharedPtr & laserCloudMsg)
{
  //  for (sensor_msgs::PointCloud2Iterator<uint16_t> it(*laserCloudMsg, "ring"); it != it.end();
  //  ++it) {
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
    imuQueue.empty() || utils->stamp2Sec(imuQueue.front().header.stamp) > timeScanCur ||
    utils->stamp2Sec(imuQueue.back().header.stamp) < timeScanEnd) {
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
  // Image Projection

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

  // Feature Matching

  cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
  cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
  copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
  copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

  kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
  kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

  laserCloudCornerLast.reset(
    new pcl::PointCloud<PointType>());  // corner feature set from odoOptimization
  laserCloudSurfLast.reset(
    new pcl::PointCloud<PointType>());  // surf feature set from odoOptimization
  laserCloudCornerLastDS.reset(
    new pcl::PointCloud<PointType>());  // downsampled corner featuer set from odoOptimization
  laserCloudSurfLastDS.reset(
    new pcl::PointCloud<PointType>());  // downsampled surf featuer set from odoOptimization

  laserCloudOri.reset(new pcl::PointCloud<PointType>());
  coeffSel.reset(new pcl::PointCloud<PointType>());

  laserCloudOriCornerVec.resize(N_SCAN_ * Horizon_SCAN_);
  coeffSelCornerVec.resize(N_SCAN_ * Horizon_SCAN_);
  laserCloudOriCornerFlag.resize(N_SCAN_ * Horizon_SCAN_);
  laserCloudOriSurfVec.resize(N_SCAN_ * Horizon_SCAN_);
  coeffSelSurfVec.resize(N_SCAN_ * Horizon_SCAN_);
  laserCloudOriSurfFlag.resize(N_SCAN_ * Horizon_SCAN_);

  std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
  std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

  laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
  laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
  laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
  laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

  kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
  kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

  for (int i = 0; i < 6; ++i) {
    transformTobeMapped[i] = 0;
  }

  matP.setZero();

  // read corner and surface map
  map_corner.reset(new pcl::PointCloud<PointType>());
  if (
    pcl::io::loadPCDFile<PointType>(
      "/home/ataparlar/data/task_specific/loam_feature_localization/bomonti_tunnel/"
      "bomonti_corner.pcd",
      *map_corner) == -1)  //* load the file
  {
    PCL_ERROR("Couldn't read corner cloud");
    PCL_ERROR("Couldn't read corner cloud");
    PCL_ERROR("Couldn't read corner cloud \n");
  }
  map_surface.reset(new pcl::PointCloud<PointType>());
  if (
    pcl::io::loadPCDFile<PointType>(
      "/home/ataparlar/data/task_specific/loam_feature_localization/bomonti_tunnel/"
      "bomonti_surface_ss.pcd",
      *map_surface) == -1)  //* load the file
  {
    PCL_ERROR("Couldn't read surface cloud");
    PCL_ERROR("Couldn't read surface cloud");
    PCL_ERROR("Couldn't read surface cloud \n");
  }
  sensor_msgs::msg::PointCloud2 ros_cloud_corner;
  pcl::toROSMsg(*map_corner, ros_cloud_corner);
  sensor_msgs::msg::PointCloud2 ros_cloud_surface;
  pcl::toROSMsg(*map_surface, ros_cloud_surface);

  ros_cloud_corner.header.frame_id = "map";
  ros_cloud_corner.header.stamp = this->get_clock()->now();
  ros_cloud_surface.header.frame_id = "map";
  ros_cloud_surface.header.stamp = this->get_clock()->now();

  pubMapCorner->publish(ros_cloud_corner);
  pubMapSurface->publish(ros_cloud_surface);

  RCLCPP_INFO(this->get_logger(), "\n\n\n\n\n\n\nPUBLISHED\n\n\n\n\n\n\n");

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
  cloudSmoothness.resize(N_SCAN_ * Horizon_SCAN_);

  downSizeFilter.setLeafSize(
    odometry_surface_leaf_size_, odometry_surface_leaf_size_, odometry_surface_leaf_size_);

  //  extractedCloud.reset(new pcl::PointCloud<PointType>());
  cornerCloud.reset(new pcl::PointCloud<PointType>());
  surfaceCloud.reset(new pcl::PointCloud<PointType>());

  cloudCurvature = new float[N_SCAN_ * Horizon_SCAN_];
  cloudNeighborPicked = new int[N_SCAN_ * Horizon_SCAN_];
  cloudLabel = new int[N_SCAN_ * Horizon_SCAN_];
}

void LoamFeatureLocalization::calculateSmoothness()
{
  int cloudSize = extractedCloud->points.size();
  for (int i = 5; i < cloudSize - 5; i++) {
    float diffRange =
      cloudInfo.point_range[i - 5] + cloudInfo.point_range[i - 4] + cloudInfo.point_range[i - 3] +
      cloudInfo.point_range[i - 2] + cloudInfo.point_range[i - 1] - cloudInfo.point_range[i] * 10 +
      cloudInfo.point_range[i + 1] + cloudInfo.point_range[i + 2] + cloudInfo.point_range[i + 3] +
      cloudInfo.point_range[i + 4] + cloudInfo.point_range[i + 5];

    cloudCurvature[i] = diffRange * diffRange;  // diffX * diffX + diffY * diffY + diffZ * diffZ;

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
  for (int i = 5; i < cloudSize - 6; ++i) {
    // occluded points
    float depth1 = cloudInfo.point_range[i];
    float depth2 = cloudInfo.point_range[i + 1];
    int columnDiff = std::abs(int(cloudInfo.point_col_index[i + 1] - cloudInfo.point_col_index[i]));
    if (columnDiff < 10) {
      // 10 pixel diff in range image
      if (depth1 - depth2 > 0.3) {
        cloudNeighborPicked[i - 5] = 1;
        cloudNeighborPicked[i - 4] = 1;
        cloudNeighborPicked[i - 3] = 1;
        cloudNeighborPicked[i - 2] = 1;
        cloudNeighborPicked[i - 1] = 1;
        cloudNeighborPicked[i] = 1;
      } else if (depth2 - depth1 > 0.3) {
        cloudNeighborPicked[i + 1] = 1;
        cloudNeighborPicked[i + 2] = 1;
        cloudNeighborPicked[i + 3] = 1;
        cloudNeighborPicked[i + 4] = 1;
        cloudNeighborPicked[i + 5] = 1;
        cloudNeighborPicked[i + 6] = 1;
      }
    }
    // parallel beam
    float diff1 = std::abs(float(cloudInfo.point_range[i - 1] - cloudInfo.point_range[i]));
    float diff2 = std::abs(float(cloudInfo.point_range[i + 1] - cloudInfo.point_range[i]));

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

  for (int i = 0; i < N_SCAN_; i++) {
    surfaceCloudScan->clear();

    for (int j = 0; j < 6; j++) {
      int sp = (cloudInfo.start_ring_index[i] * (6 - j) + cloudInfo.end_ring_index[i] * j) / 6;
      int ep =
        (cloudInfo.start_ring_index[i] * (5 - j) + cloudInfo.end_ring_index[i] * (j + 1)) / 6 - 1;

      if (sp >= ep) continue;

      std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, by_value());

      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {
        int ind = cloudSmoothness[k].ind;
        if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edge_threshold_) {
          largestPickedNum++;
          if (largestPickedNum <= 20) {
            cloudLabel[ind] = 1;
            cornerCloud->push_back(extractedCloud->points[ind]);
          } else {
            break;
          }

          cloudNeighborPicked[ind] = 1;
          for (int l = 1; l <= 5; l++) {
            int columnDiff = std::abs(
              int(cloudInfo.point_col_index[ind + l] - cloudInfo.point_col_index[ind + l - 1]));
            if (columnDiff > 10) break;
            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            int columnDiff = std::abs(
              int(cloudInfo.point_col_index[ind + l] - cloudInfo.point_col_index[ind + l + 1]));
            if (columnDiff > 10) break;
            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      for (int k = sp; k <= ep; k++) {
        int ind = cloudSmoothness[k].ind;
        if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surface_threshold_) {
          cloudLabel[ind] = -1;
          cloudNeighborPicked[ind] = 1;

          for (int l = 1; l <= 5; l++) {
            int columnDiff = std::abs(
              int(cloudInfo.point_col_index[ind + l] - cloudInfo.point_col_index[ind + l - 1]));
            if (columnDiff > 10) break;

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            int columnDiff = std::abs(
              int(cloudInfo.point_col_index[ind + l] - cloudInfo.point_col_index[ind + l + 1]));
            if (columnDiff > 10) break;

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      for (int k = sp; k <= ep; k++) {
        if (cloudLabel[k] <= 0) {
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

// void LoamFeatureLocalization::publishFeatureCloud()
//{
//   // free cloud info memory
//   freeCloudInfoMemory();
//   // save newly extracted features
//   cloudInfo.cloud_corner = publishCloud(pubCornerPoints,  cornerCloud,  cloudHeader.stamp,
//   lidarFrame); cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud,
//   cloudHeader.stamp, lidarFrame);
//   // publish to mapOptimization
//   pubLaserCloudInfo->publish(cloudInfo);
// }

// Feature Matching

void LoamFeatureLocalization::updateInitialGuess()
{
  // save current transformation before any processing
  incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);  // lidar tf ?

  std::cout << "initial guess 1" << std::endl;

  std::cout << "cloudKeyPoses3D->points.size(): " << cloudKeyPoses3D->points.size() << std::endl;

  static Eigen::Affine3f lastImuTransformation;
  // initialization
  if (cloudKeyPoses3D->points.empty()) {
    std::cout << "a" << std::endl;
    transformTobeMapped[0] = cloudInfo.imu_roll_init;
    transformTobeMapped[1] = cloudInfo.imu_pitch_init;
    transformTobeMapped[2] = cloudInfo.imu_yaw_init;
    std::cout << "b" << std::endl;

    //    if (!useImuHeadingInitialization)
    //      transformTobeMapped[2] = 0;

    lastImuTransformation = pcl::getTransformation(
      0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init,
      cloudInfo.imu_yaw_init);  // save imu before return;
    std::cout << "c" << std::endl;
    //    return;
  }
  std::cout << "initial guess 2" << std::endl;

  // use imu pre-integration estimation for pose guess
  static bool lastImuPreTransAvailable = false;
  static Eigen::Affine3f lastImuPreTransformation;
  if (cloudInfo.odom_available == true) {
    Eigen::Affine3f transBack = pcl::getTransformation(
      cloudInfo.initial_guess_x, cloudInfo.initial_guess_y, cloudInfo.initial_guess_z,
      cloudInfo.initial_guess_roll, cloudInfo.initial_guess_pitch, cloudInfo.initial_guess_yaw);
    if (lastImuPreTransAvailable == false) {
      lastImuPreTransformation = transBack;
      lastImuPreTransAvailable = true;
    } else {
      Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
      Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
      Eigen::Affine3f transFinal = transTobe * transIncre;
      pcl::getTranslationAndEulerAngles(
        transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

      lastImuPreTransformation = transBack;

      lastImuTransformation = pcl::getTransformation(
        0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init,
        cloudInfo.imu_yaw_init);  // save imu before return;
      return;
    }
  }
  std::cout << "initial guess 3" << std::endl;

  // use imu incremental estimation for pose guess (only rotation)
  if (cloudInfo.imu_available == true) {
    Eigen::Affine3f transBack = pcl::getTransformation(
      0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init);
    Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

    Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
    Eigen::Affine3f transFinal = transTobe * transIncre;
    pcl::getTranslationAndEulerAngles(
      transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
      transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

    lastImuTransformation = pcl::getTransformation(
      0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init,
      cloudInfo.imu_yaw_init);  // save imu before return;
    return;
  }
  std::cout << "initial guess 4" << std::endl;
}

void LoamFeatureLocalization::extractSurroundingKeyFrames()
{
  if (cloudKeyPoses3D->points.empty() == true) return;

  // if (loopClosureEnableFlag == true)
  // {
  //     extractForLoopClosure();
  // } else {
  //     extractNearby();
  // }
  std::cout << "buraya geldi mi ?" << std::endl;
  extractNearby();
}

void LoamFeatureLocalization::extractNearby()
{
  pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
  std::vector<int> pointSearchInd;
  std::vector<float> pointSearchSqDis;

  // extract all the nearby key poses and downsample them
  kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);  // create kd-tree
  kdtreeSurroundingKeyPoses->radiusSearch(
    cloudKeyPoses3D->back(), (double)surrounding_key_frame_search_radius_, pointSearchInd,
    pointSearchSqDis);
  for (int i = 0; i < (int)pointSearchInd.size(); ++i) {
    int id = pointSearchInd[i];
    surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
  }

  downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
  downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
  for (auto & pt : surroundingKeyPosesDS->points) {
    kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
    pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
  }

  // also extract some latest key frames in case the robot rotates in one position
  int numPoses = cloudKeyPoses3D->size();
  for (int i = numPoses - 1; i >= 0; --i) {
    if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
      surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
    else
      break;
  }

  extractCloud(surroundingKeyPosesDS);
}

void LoamFeatureLocalization::extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
{
  // fuse the map
  laserCloudCornerFromMap->clear();
  laserCloudSurfFromMap->clear();
  for (int i = 0; i < (int)cloudToExtract->size(); ++i) {
    if (
      utils->pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) >
      surrounding_key_frame_search_radius_)
      continue;

    int thisKeyInd = (int)cloudToExtract->points[i].intensity;
    if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) {
      // transformed cloud available
      *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
      *laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second;
    } else {
      // transformed cloud not available
      pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(
        cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
      pcl::PointCloud<PointType> laserCloudSurfTemp =
        *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
      *laserCloudCornerFromMap += laserCloudCornerTemp;
      *laserCloudSurfFromMap += laserCloudSurfTemp;
      laserCloudMapContainer[thisKeyInd] = std::make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
    }
  }

  // Downsample the surrounding corner key frames (or map)
  //        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
  downSizeFilterCorner.setInputCloud(map_corner);
  downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
  laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
  // Downsample the surrounding surf key frames (or map)
  //        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
  downSizeFilterSurf.setInputCloud(map_surface);
  downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
  laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

  // clear map cache if too large
  if (laserCloudMapContainer.size() > 1000) laserCloudMapContainer.clear();
}

pcl::PointCloud<PointType>::Ptr LoamFeatureLocalization::transformPointCloud(
  pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose * transformIn)
{
  pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

  int cloudSize = cloudIn->size();
  cloudOut->resize(cloudSize);

  Eigen::Affine3f transCur = pcl::getTransformation(
    transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch,
    transformIn->yaw);

#pragma omp parallel for num_threads(numberOfCores)
  for (int i = 0; i < cloudSize; ++i) {
    const auto & pointFrom = cloudIn->points[i];
    cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y +
                            transCur(0, 2) * pointFrom.z + transCur(0, 3);
    cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y +
                            transCur(1, 2) * pointFrom.z + transCur(1, 3);
    cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y +
                            transCur(2, 2) * pointFrom.z + transCur(2, 3);
    cloudOut->points[i].intensity = pointFrom.intensity;
  }
  return cloudOut;
}

void LoamFeatureLocalization::downsampleCurrentScan()
{
  // Downsample cloud from current scan
  laserCloudCornerLastDS->clear();
  downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
  downSizeFilterCorner.filter(*laserCloudCornerLastDS);
  laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

  laserCloudSurfLastDS->clear();
  downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
  downSizeFilterSurf.filter(*laserCloudSurfLastDS);
  laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
}

void LoamFeatureLocalization::scan2MapOptimization()
{
  if (cloudKeyPoses3D->points.empty()) return;

  if (
    laserCloudCornerLastDSNum > edge_feature_min_valid_num_ &&
    laserCloudSurfLastDSNum > surf_feature_min_valid_num_) {
    kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
    kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
    //            kdtreeCornerFromMap->setInputCloud(map_corner);
    //            kdtreeSurfFromMap->setInputCloud(map_surface);

    for (int iterCount = 0; iterCount < 30; iterCount++) {
      laserCloudOri->clear();
      coeffSel->clear();

      cornerOptimization();
      surfOptimization();

      combineOptimizationCoeffs();

      if (LMOptimization(iterCount) == true) break;
    }

    transformUpdate();
  } else {
    RCLCPP_WARN(
      get_logger(), "Not enough features! Only %d edge and %d planar features available.",
      laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
  }
}

void LoamFeatureLocalization::cornerOptimization()
{
  updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
  for (int i = 0; i < laserCloudCornerLastDSNum; i++) {
    PointType pointOri, pointSel, coeff;
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    pointOri = laserCloudCornerLastDS->points[i];
    pointAssociateToMap(&pointOri, &pointSel);
    kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

    cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

    if (pointSearchSqDis[4] < 1.0) {
      float cx = 0, cy = 0, cz = 0;
      for (int j = 0; j < 5; j++) {
        cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
        cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
        cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
      }
      cx /= 5;
      cy /= 5;
      cz /= 5;

      float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
      for (int j = 0; j < 5; j++) {
        float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
        float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
        float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

        a11 += ax * ax;
        a12 += ax * ay;
        a13 += ax * az;
        a22 += ay * ay;
        a23 += ay * az;
        a33 += az * az;
      }
      a11 /= 5;
      a12 /= 5;
      a13 /= 5;
      a22 /= 5;
      a23 /= 5;
      a33 /= 5;

      matA1.at<float>(0, 0) = a11;
      matA1.at<float>(0, 1) = a12;
      matA1.at<float>(0, 2) = a13;
      matA1.at<float>(1, 0) = a12;
      matA1.at<float>(1, 1) = a22;
      matA1.at<float>(1, 2) = a23;
      matA1.at<float>(2, 0) = a13;
      matA1.at<float>(2, 1) = a23;
      matA1.at<float>(2, 2) = a33;

      cv::eigen(matA1, matD1, matV1);

      if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
        float x0 = pointSel.x;
        float y0 = pointSel.y;
        float z0 = pointSel.z;
        float x1 = cx + 0.1 * matV1.at<float>(0, 0);
        float y1 = cy + 0.1 * matV1.at<float>(0, 1);
        float z1 = cz + 0.1 * matV1.at<float>(0, 2);
        float x2 = cx - 0.1 * matV1.at<float>(0, 0);
        float y2 = cy - 0.1 * matV1.at<float>(0, 1);
        float z2 = cz - 0.1 * matV1.at<float>(0, 2);

        float a012 = sqrt(
          ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) *
            ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
          ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) *
            ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
          ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) *
            ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

        float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

        float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                    (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) /
                   a012 / l12;

        float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) -
                     (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
                   a012 / l12;

        float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                     (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
                   a012 / l12;

        float ld2 = a012 / l12;

        float s = 1 - 0.9 * fabs(ld2);

        coeff.x = s * la;
        coeff.y = s * lb;
        coeff.z = s * lc;
        coeff.intensity = s * ld2;

        if (s > 0.1) {
          laserCloudOriCornerVec[i] = pointOri;
          coeffSelCornerVec[i] = coeff;
          laserCloudOriCornerFlag[i] = true;
        }
      }
    }
  }
}

void LoamFeatureLocalization::surfOptimization()
{
  updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
  for (int i = 0; i < laserCloudSurfLastDSNum; i++) {
    PointType pointOri, pointSel, coeff;
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    pointOri = laserCloudSurfLastDS->points[i];
    pointAssociateToMap(&pointOri, &pointSel);
    kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

    Eigen::Matrix<float, 5, 3> matA0;
    Eigen::Matrix<float, 5, 1> matB0;
    Eigen::Vector3f matX0;

    matA0.setZero();
    matB0.fill(-1);
    matX0.setZero();

    if (pointSearchSqDis[4] < 1.0) {
      for (int j = 0; j < 5; j++) {
        matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
        matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
        matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
      }

      matX0 = matA0.colPivHouseholderQr().solve(matB0);

      float pa = matX0(0, 0);
      float pb = matX0(1, 0);
      float pc = matX0(2, 0);
      float pd = 1;

      float ps = sqrt(pa * pa + pb * pb + pc * pc);
      pa /= ps;
      pb /= ps;
      pc /= ps;
      pd /= ps;

      bool planeValid = true;
      for (int j = 0; j < 5; j++) {
        if (
          fabs(
            pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
            pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
            pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
          planeValid = false;
          break;
        }
      }

      if (planeValid) {
        float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

        float s =
          1 -
          0.9 * fabs(pd2) /
            sqrt(sqrt(pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

        coeff.x = s * pa;
        coeff.y = s * pb;
        coeff.z = s * pc;
        coeff.intensity = s * pd2;

        if (s > 0.1) {
          laserCloudOriSurfVec[i] = pointOri;
          coeffSelSurfVec[i] = coeff;
          laserCloudOriSurfFlag[i] = true;
        }
      }
    }
  }
}

void LoamFeatureLocalization::combineOptimizationCoeffs()
{
  // combine corner coeffs
  for (int i = 0; i < laserCloudCornerLastDSNum; ++i) {
    if (laserCloudOriCornerFlag[i] == true) {
      laserCloudOri->push_back(laserCloudOriCornerVec[i]);
      coeffSel->push_back(coeffSelCornerVec[i]);
    }
  }
  // combine surf coeffs
  for (int i = 0; i < laserCloudSurfLastDSNum; ++i) {
    if (laserCloudOriSurfFlag[i] == true) {
      laserCloudOri->push_back(laserCloudOriSurfVec[i]);
      coeffSel->push_back(coeffSelSurfVec[i]);
    }
  }
  // reset flag for next iteration
  std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
  std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
}

bool LoamFeatureLocalization::LMOptimization(int iterCount)
{
  // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate
  // transformation lidar <- camera      ---     camera <- lidar x = z                ---     x = y
  // y = x                ---     y = z
  // z = y                ---     z = x
  // roll = yaw           ---     roll = pitch
  // pitch = roll         ---     pitch = yaw
  // yaw = pitch          ---     yaw = roll

  // lidar -> camera
  float srx = sin(transformTobeMapped[1]);
  float crx = cos(transformTobeMapped[1]);
  float sry = sin(transformTobeMapped[2]);
  float cry = cos(transformTobeMapped[2]);
  float srz = sin(transformTobeMapped[0]);
  float crz = cos(transformTobeMapped[0]);

  int laserCloudSelNum = laserCloudOri->size();
  if (laserCloudSelNum < 50) {
    return false;
  }

  cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
  cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
  cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
  cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
  cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
  cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
  cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

  PointType pointOri, coeff;

  for (int i = 0; i < laserCloudSelNum; i++) {
    // lidar -> camera
    pointOri.x = laserCloudOri->points[i].y;
    pointOri.y = laserCloudOri->points[i].z;
    pointOri.z = laserCloudOri->points[i].x;
    // lidar -> camera
    coeff.x = coeffSel->points[i].y;
    coeff.y = coeffSel->points[i].z;
    coeff.z = coeffSel->points[i].x;
    coeff.intensity = coeffSel->points[i].intensity;
    // in camera
    float arx =
      (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y - srx * sry * pointOri.z) *
        coeff.x +
      (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.y +
      (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) *
        coeff.z;

    float ary = ((cry * srx * srz - crz * sry) * pointOri.x +
                 (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) *
                  coeff.x +
                ((-cry * crz - srx * sry * srz) * pointOri.x +
                 (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) *
                  coeff.z;

    float arz =
      ((crz * srx * sry - cry * srz) * pointOri.x + (-cry * crz - srx * sry * srz) * pointOri.y) *
        coeff.x +
      (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y +
      ((sry * srz + cry * crz * srx) * pointOri.x + (crz * sry - cry * srx * srz) * pointOri.y) *
        coeff.z;
    // lidar -> camera
    matA.at<float>(i, 0) = arz;
    matA.at<float>(i, 1) = arx;
    matA.at<float>(i, 2) = ary;
    matA.at<float>(i, 3) = coeff.z;
    matA.at<float>(i, 4) = coeff.x;
    matA.at<float>(i, 5) = coeff.y;
    matB.at<float>(i, 0) = -coeff.intensity;
  }

  cv::transpose(matA, matAt);
  matAtA = matAt * matA;
  matAtB = matAt * matB;
  cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

  if (iterCount == 0) {
    cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

    cv::eigen(matAtA, matE, matV);
    matV.copyTo(matV2);

    isDegenerate = false;
    float eignThre[6] = {100, 100, 100, 100, 100, 100};
    for (int i = 5; i >= 0; i--) {
      if (matE.at<float>(0, i) < eignThre[i]) {
        for (int j = 0; j < 6; j++) {
          matV2.at<float>(i, j) = 0;
        }
        isDegenerate = true;
      } else {
        break;
      }
    }
    matP = matV.inv() * matV2;
  }

  if (isDegenerate) {
    cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
    matX.copyTo(matX2);
    matX = matP * matX2;
  }

  transformTobeMapped[0] += matX.at<float>(0, 0);
  transformTobeMapped[1] += matX.at<float>(1, 0);
  transformTobeMapped[2] += matX.at<float>(2, 0);
  transformTobeMapped[3] += matX.at<float>(3, 0);
  transformTobeMapped[4] += matX.at<float>(4, 0);
  transformTobeMapped[5] += matX.at<float>(5, 0);

  float deltaR = sqrt(
    pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) + pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
    pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
  float deltaT = sqrt(
    pow(matX.at<float>(3, 0) * 100, 2) + pow(matX.at<float>(4, 0) * 100, 2) +
    pow(matX.at<float>(5, 0) * 100, 2));

  if (deltaR < 0.05 && deltaT < 0.05) {
    return true;  // converged
  }
  return false;  // keep optimizing
}

void LoamFeatureLocalization::transformUpdate()
{
  if (cloudInfo.imu_available == true) {
    if (std::abs(cloudInfo.imu_pitch_init) < 1.4) {
      double imuWeight = imu_rpy_weight_;
      tf2::Quaternion imuQuaternion;
      tf2::Quaternion transformQuaternion;
      double rollMid, pitchMid, yawMid;

      // slerp roll
      transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
      imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
      tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
        .getRPY(rollMid, pitchMid, yawMid);
      transformTobeMapped[0] = rollMid;

      // slerp pitch
      transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
      imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
      tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
        .getRPY(rollMid, pitchMid, yawMid);
      transformTobeMapped[1] = pitchMid;
    }
  }

  transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance_);
  transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance_);
  transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance_);

  incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
}

void LoamFeatureLocalization::updatePointAssociateToMap()
{
  transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
}

Eigen::Affine3f LoamFeatureLocalization::trans2Affine3f(float transformIn[])
{
  return pcl::getTransformation(
    transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}

void LoamFeatureLocalization::pointAssociateToMap(PointType const * const pi, PointType * const po)
{
  po->x = transPointAssociateToMap(0, 0) * pi->x + transPointAssociateToMap(0, 1) * pi->y +
          transPointAssociateToMap(0, 2) * pi->z + transPointAssociateToMap(0, 3);
  po->y = transPointAssociateToMap(1, 0) * pi->x + transPointAssociateToMap(1, 1) * pi->y +
          transPointAssociateToMap(1, 2) * pi->z + transPointAssociateToMap(1, 3);
  po->z = transPointAssociateToMap(2, 0) * pi->x + transPointAssociateToMap(2, 1) * pi->y +
          transPointAssociateToMap(2, 2) * pi->z + transPointAssociateToMap(2, 3);
  po->intensity = pi->intensity;
}

float LoamFeatureLocalization::constraintTransformation(float value, float limit)
{
  if (value < -limit) value = -limit;
  if (value > limit) value = limit;

  return value;
}

void LoamFeatureLocalization::saveKeyFramesAndFactor()
{
  if (saveFrame() == false) return;

  //    // odom factor
  addOdomFactor();
  //
  //    // gps factor
  //    addGPSFactor();
  //
  //    // loop factor
  //    addLoopFactor();

  // cout << "****************************************************" << endl;
  // gtSAMgraph.print("GTSAM Graph:\n");

  // update iSAM

  std::cout << "saveKeyFramesAndFactor 1" << std::endl;

  std::cout << "initialEstimate.size(): " << initialEstimate.size() << std::endl;
  isam->update(gtSAMgraph, initialEstimate);
  isam->update();

  if (aLoopIsClosed == true) {
    isam->update();
    isam->update();
    isam->update();
    isam->update();
    isam->update();
  }

  std::cout << "saveKeyFramesAndFactor 2" << std::endl;

  gtSAMgraph.resize(0);
  initialEstimate.clear();
  std::cout << "saveKeyFramesAndFactor 2 - 1" << std::endl;

  // save key poses
  PointType thisPose3D;
  PointTypePose thisPose6D;
  gtsam::Pose3 latestEstimate;

  std::cout << "saveKeyFramesAndFactor 2 - 2" << std::endl;
  isamCurrentEstimate = isam->calculateEstimate();
  std::cout << "saveKeyFramesAndFactor 2 - 3" << std::endl;
  std::cout << "isamCurrentEstimate.size(): " << isamCurrentEstimate.size() << std::endl;
  latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size() - 1);
  // cout << "****************************************************" << endl;
  // isamCurrentEstimate.print("Current estimate: ");

  std::cout << "saveKeyFramesAndFactor 3" << std::endl;

  thisPose3D.x = latestEstimate.translation().x();
  thisPose3D.y = latestEstimate.translation().y();
  thisPose3D.z = latestEstimate.translation().z();
  thisPose3D.intensity = cloudKeyPoses3D->size();  // this can be used as index
  cloudKeyPoses3D->push_back(thisPose3D);

  thisPose6D.x = thisPose3D.x;
  thisPose6D.y = thisPose3D.y;
  thisPose6D.z = thisPose3D.z;
  thisPose6D.intensity = thisPose3D.intensity;  // this can be used as index
  thisPose6D.roll = latestEstimate.rotation().roll();
  thisPose6D.pitch = latestEstimate.rotation().pitch();
  thisPose6D.yaw = latestEstimate.rotation().yaw();
  thisPose6D.time = timeLaserInfoCur;
  cloudKeyPoses6D->push_back(thisPose6D);

  // cout << "****************************************************" << endl;
  // cout << "Pose covariance:" << endl;
  // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
  poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

  // save updated transform
  transformTobeMapped[0] = latestEstimate.rotation().roll();
  transformTobeMapped[1] = latestEstimate.rotation().pitch();
  transformTobeMapped[2] = latestEstimate.rotation().yaw();
  transformTobeMapped[3] = latestEstimate.translation().x();
  transformTobeMapped[4] = latestEstimate.translation().y();
  transformTobeMapped[5] = latestEstimate.translation().z();

  // save all the received edge and surf points
  pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
  pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
  pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);

  // save key frame cloud
  cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
  surfCloudKeyFrames.push_back(thisSurfKeyFrame);

  std::cout << "saveKeyFramesAndFactor 4" << std::endl;

  // save path for visualization
  updatePath(thisPose6D);
  std::cout << "saveKeyFramesAndFactor 5" << std::endl;
}

bool LoamFeatureLocalization::saveFrame()
{
  if (cloudKeyPoses3D->points.empty()) return true;

  //    if (sensor == SensorType::LIVOX)
  //    {
  //      if (timeLaserInfoCur - cloudKeyPoses6D->back().time > 1.0)
  //        return true;
  //    }

  Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
  Eigen::Affine3f transFinal = pcl::getTransformation(
    transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], transformTobeMapped[0],
    transformTobeMapped[1], transformTobeMapped[2]);
  Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
  float x, y, z, roll, pitch, yaw;
  pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

  if (
    abs(roll) < surrounding_key_frame_adding_angle_threshold_ &&
    abs(pitch) < surrounding_key_frame_adding_angle_threshold_ &&
    abs(yaw) < surrounding_key_frame_adding_angle_threshold_ &&
    sqrt(x * x + y * y + z * z) < surrounding_key_frame_adding_dist_threshold_)
    return false;

  return true;
}

Eigen::Affine3f LoamFeatureLocalization::pclPointToAffine3f(PointTypePose thisPoint)
{
  return pcl::getTransformation(
    thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

void LoamFeatureLocalization::updatePath(const PointTypePose & pose_in)
{
  geometry_msgs::msg::PoseStamped pose_stamped;
  pose_stamped.header.stamp = rclcpp::Time(pose_in.time * 1e9);
  pose_stamped.header.frame_id = output_odometry_frame_;
  pose_stamped.pose.position.x = pose_in.x;
  pose_stamped.pose.position.y = pose_in.y;
  pose_stamped.pose.position.z = pose_in.z;
  tf2::Quaternion q;
  q.setRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
  pose_stamped.pose.orientation.x = q.x();
  pose_stamped.pose.orientation.y = q.y();
  pose_stamped.pose.orientation.z = q.z();
  pose_stamped.pose.orientation.w = q.w();

  globalPath.poses.push_back(pose_stamped);
}

void LoamFeatureLocalization::correctPoses()
{
  if (cloudKeyPoses3D->points.empty()) return;

  if (aLoopIsClosed == true) {
    // clear map cache
    laserCloudMapContainer.clear();
    // clear path
    globalPath.poses.clear();
    // update key poses
    int numPoses = isamCurrentEstimate.size();
    for (int i = 0; i < numPoses; ++i) {
      cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
      cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
      cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

      cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
      cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
      cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
      cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
      cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
      cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

      updatePath(cloudKeyPoses6D->points[i]);
    }

    aLoopIsClosed = false;
  }
}

void LoamFeatureLocalization::publishOdometry()
{
  // Publish odometry for ROS (global)
  nav_msgs::msg::Odometry laserOdometryROS;
  laserOdometryROS.header.stamp = timeLaserInfoStamp;
  laserOdometryROS.header.frame_id = "map";
  laserOdometryROS.child_frame_id = output_odometry_frame_;
  laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
  laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
  laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
  tf2::Quaternion quat_tf;
  quat_tf.setRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
  geometry_msgs::msg::Quaternion quat_msg;
  tf2::convert(quat_tf, quat_msg);
  laserOdometryROS.pose.pose.orientation = quat_msg;
  pubLaserOdometryGlobal->publish(laserOdometryROS);

  // Publish TF
  quat_tf.setRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
  tf2::Transform t_odom_to_lidar = tf2::Transform(
    quat_tf, tf2::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
  tf2::TimePoint time_point = tf2_ros::fromRclcpp(timeLaserInfoStamp);
  tf2::Stamped<tf2::Transform> temp_odom_to_lidar(
    t_odom_to_lidar, time_point, output_odometry_frame_);
  geometry_msgs::msg::TransformStamped trans_odom_to_lidar;
  tf2::convert(temp_odom_to_lidar, trans_odom_to_lidar);
  trans_odom_to_lidar.header.frame_id = "map";
  trans_odom_to_lidar.child_frame_id = output_odometry_frame_;
  br->sendTransform(trans_odom_to_lidar);
}

void LoamFeatureLocalization::addOdomFactor()
{
  if (cloudKeyPoses3D->points.empty()) {
    gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances(
      (gtsam::Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8)
        .finished());  // rad*rad, meter*meter
    gtSAMgraph.add(
      gtsam::PriorFactor<gtsam::Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
    initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
  } else {
    gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(
      (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
    gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);
    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(
      cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo),
      odometryNoise));
    initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
  }
}

gtsam::Pose3 LoamFeatureLocalization::trans2gtsamPose(float transformIn[])
{
  return gtsam::Pose3(
    gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
    gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

gtsam::Pose3 LoamFeatureLocalization::pclPointTogtsamPose3(PointTypePose thisPoint)
{
  return gtsam::Pose3(
    gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
    gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
}

}  // namespace loam_feature_localization

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(loam_feature_localization::LoamFeatureLocalization)
