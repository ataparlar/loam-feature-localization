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

#include "loam_feature_localization/utils.hpp"


#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace loam_feature_localization
{

Utils::Utils(double lidar_imu_roll_, double lidar_imu_pitch_, double lidar_imu_yaw_,
             double lidar_imu_x_, double lidar_imu_y_, double lidar_imu_z_)
{
    Eigen::Matrix3d matrix;
    matrix.topLeftCorner<3, 3>() =
      Eigen::AngleAxisd(deg_to_rad(lidar_imu_yaw_), Eigen::Vector3d::UnitZ()).toRotationMatrix() *
      Eigen::AngleAxisd(deg_to_rad(lidar_imu_pitch_), Eigen::Vector3d::UnitY()).toRotationMatrix() *
      Eigen::AngleAxisd(deg_to_rad(lidar_imu_roll_), Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Vector3d vector(lidar_imu_x_, lidar_imu_y_, lidar_imu_z_);
    Eigen::Quaterniond quat(matrix);

    extRot = matrix;
    extRPY = matrix;
    extTrans = vector;
    extQRPY = quat;
}

std::string Utils::byte_hex_to_string(uint8_t byte_hex)
{
  std::stringstream ss;
  ss << std::hex << std::setfill('0');
  ss << "data_packet_with_header: " << std::setw(2) << (int)byte_hex;
  return ss.str();
}

std::string Utils::bytes_hexes_to_string(const std::vector<uint8_t> & bytes_hexes)
{
  std::string output;
  for (const auto & byte_hex : bytes_hexes) {
    output += byte_hex_to_string(byte_hex) + " ";
  }
  if (output.empty()) {
    throw std::length_error("output.empty()");
  }
  output.erase(output.end() - 1, output.end());
  return output;
}

sensor_msgs::msg::Imu Utils::imuConverter(const sensor_msgs::msg::Imu& imu_in)
{
  sensor_msgs::msg::Imu imu_out = imu_in;
  // rotate acceleration
  Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
  acc = extRot * acc;
  imu_out.linear_acceleration.x = acc.x();
  imu_out.linear_acceleration.y = acc.y();
  imu_out.linear_acceleration.z = acc.z();
  // rotate gyroscope
  Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
  gyr = extRot * gyr;
  imu_out.angular_velocity.x = gyr.x();
  imu_out.angular_velocity.y = gyr.y();
  imu_out.angular_velocity.z = gyr.z();
  // rotate roll pitch yaw
  Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
  Eigen::Quaterniond q_final = q_from * extQRPY;
  imu_out.orientation.x = q_final.x();
  imu_out.orientation.y = q_final.y();
  imu_out.orientation.z = q_final.z();
  imu_out.orientation.w = q_final.w();

  if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
  {
    RCLCPP_ERROR(rclcpp::get_logger("error"), "Invalid quaternion, please use a 9-axis IMU!");
    rclcpp::shutdown();
  }

  return imu_out;
}

//template<typename T>
void Utils::imuAngular2rosAngular(sensor_msgs::msg::Imu *thisImuMsg, double *angular_x, double *angular_y, double *angular_z)
{
  *angular_x = thisImuMsg->angular_velocity.x;
  *angular_y = thisImuMsg->angular_velocity.y;
  *angular_z = thisImuMsg->angular_velocity.z;
}

//template<typename T>
void Utils::imuRPY2rosRPY(sensor_msgs::msg::Imu *thisImuMsg, double *rosRoll, double *rosPitch, double *rosYaw)
{
  double imuRoll, imuPitch, imuYaw;
  tf2::Quaternion orientation;
  tf2::fromMsg(thisImuMsg->orientation, orientation);
  tf2::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

  *rosRoll = imuRoll;
  *rosPitch = imuPitch;
  *rosYaw = imuYaw;
}

float Utils::pointDistance(PointType p)
{
  return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}
float Utils::pointDistance(PointType p1, PointType p2)
{
  return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}


//std::vector<std::string> Utils::string_to_vec_split_by(const std::string & input, char splitter)
//{
//  std::stringstream ss_input(input);
//  std::string segment;
//  std::vector<std::string> seglist;
//  while (std::getline(ss_input, segment, splitter)) {
//    seglist.push_back(segment);
//  }
//  return seglist;
//}
//
//Eigen::Matrix3d Utils::ned2enu_converter_for_matrices(const Eigen::Matrix3d & matrix3d)
//{
//  Eigen::Matrix3d ned2enu;
//  ned2enu.matrix().topLeftCorner<3, 3>() =
//    Eigen::AngleAxisd(Utils::deg_to_rad(-90.0), Eigen::Vector3d::UnitZ())
//      .toRotationMatrix() *
//    Eigen::AngleAxisd(Utils::deg_to_rad(0.0), Eigen::Vector3d::UnitY())
//      .toRotationMatrix() *
//    Eigen::AngleAxisd(Utils::deg_to_rad(180.0), Eigen::Vector3d::UnitX())
//      .toRotationMatrix();
//
//  Eigen::Matrix3d output_matrix;
//  output_matrix = matrix3d.matrix() * ned2enu.matrix();
//
//  return output_matrix;
//}

}  // namespace loam_mapper::utils