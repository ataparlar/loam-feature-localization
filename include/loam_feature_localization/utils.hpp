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

#ifndef LOAM_FEATURE_LOCALIZATION__UTILS_HPP_
#define LOAM_FEATURE_LOCALIZATION__UTILS_HPP_

#include <Eigen/Geometry>
#include <pcl/impl/point_types.hpp>
#include <rclcpp/rclcpp.hpp>

#include "sensor_msgs/msg/imu.hpp"

#include <pcl/register_point_struct.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

struct VelodynePointXYZIRT
{
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY
  uint8_t return_type;
  uint16_t channel;
  float azimuth;
  float elevation;
  float distance;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(
  VelodynePointXYZIRT,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
    uint16_t, return_type, return_type)(uint16_t, channel, channel)(float, azimuth, azimuth)(
    float, elevation, elevation)(float, distance, distance)(float, time, time))


struct OusterPointXYZIRT
{
  PCL_ADD_POINT4D;
  float intensity;
  uint32_t t;
  uint16_t reflectivity;
  uint8_t ring;
  uint16_t noise;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(
  OusterPointXYZIRT,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint32_t, t, t)(
    uint16_t, reflectivity,
    reflectivity)(uint8_t, ring, ring)(uint16_t, noise, noise)(uint32_t, range, range))

using PointXYZIRT = VelodynePointXYZIRT;

/*
 * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is
 * time stamp)
 */
struct PointXYZIRPYT
{
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY  // preferred way of adding a XYZ+padding
    float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
} EIGEN_ALIGN16;                   // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(
  PointXYZIRPYT, (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
                   float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))

typedef PointXYZIRPYT PointTypePose;

namespace loam_feature_localization
{

typedef pcl::PointXYZI PointType;

class Utils
{
public:
  using SharedPtr = std::shared_ptr<Utils>;
  using ConstSharedPtr = const SharedPtr;

//  explicit Utils(
//    double lidar_imu_roll_, double lidar_imu_pitch_, double lidar_imu_yaw_, double lidar_imu_x_,
//    double lidar_imu_y_, double lidar_imu_z_);
  explicit Utils(Eigen::Matrix3d ext_rot,
                 Eigen::Matrix3d ext_rpy,
                 Eigen::Vector3d ext_trans);

  std::string byte_hex_to_string(uint8_t byte_hex);
  std::string bytes_hexes_to_string(const std::vector<uint8_t> & bytes_hexes);
  sensor_msgs::msg::Imu imuConverter(const sensor_msgs::msg::Imu & imu_in);
  //  static std::vector<std::string> string_to_vec_split_by(const std::string &
  //  input, char splitter); static Eigen::Matrix3d
  //  ned2enu_converter_for_matrices(const Eigen::Matrix3d & matrix3d);

  Eigen::Matrix3d ext_rot_;
  Eigen::Matrix3d ext_rpy_;
  Eigen::Vector3d ext_trans_;
  Eigen::Quaterniond ext_qrpy_;

  template <typename T>
  T deg_to_rad(T deg)
  {
    constexpr double multiplier = M_PI / 180.0;
    return static_cast<T>(deg * multiplier);
  }

  template <typename T>
  double stamp2Sec(const T & stamp)
  {
    return rclcpp::Time(stamp).seconds();
  }

  rclcpp::Time get_time() { return rclcpp::Clock().now(); }

  //  template<typename T>
  void imuAngular2rosAngular(
    sensor_msgs::msg::Imu * thisImuMsg, double * angular_x, double * angular_y, double * angular_z);

  //  template<typename T>
  void imuRPY2rosRPY(
    sensor_msgs::msg::Imu * thisImuMsg, double * rosRoll, double * rosPitch, double * rosYaw);
  float pointDistance(PointType p);
  float pointDistance(PointType p1, PointType p2);

  struct CloudInfo
  {
    bool imu_available;
    bool odom_available;
    double imu_roll_init;
    double imu_pitch_init;
    double imu_yaw_init;
    double initial_guess_x;
    double initial_guess_y;
    double initial_guess_z;
    double initial_guess_roll;
    double initial_guess_pitch;
    double initial_guess_yaw;
    std::vector<float> point_range;
    std::vector<uint32_t> start_ring_index;
    std::vector<uint32_t> point_col_index;
    std::vector<int32_t> end_ring_index;
  };
};

}  // namespace loam_feature_localization

#endif  // LOAM_FEATURE_LOCALIZATION__UTILS_HPP_
