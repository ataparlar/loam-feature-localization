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

#include "loam_feature_localization/feature_matching.hpp"

#include <pcl/common/eigen.h>
#include <pcl/common/angles.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>


namespace loam_feature_localization
{
FeatureMatching::FeatureMatching(
  int n_scan, int horizon_scan, std::string corner_map_path, std::string surface_map_path,
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_map_corner,
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_map_surface,
  rclcpp::Time now, const Utils::SharedPtr & utils,
  const std::string & odometry_frame,  const std::string & base_link_frame,
  double surrounding_key_frame_search_radius, double surrounding_key_frame_adding_angle_threshold,
  double surrounding_key_frame_adding_dist_threshold, double surrounding_key_frame_density,
  double mapping_corner_leaf_size, double mapping_surface_leaf_size,
  int min_edge_feature_number, int min_surface_feature_number,
  double rotation_tollerance, double z_tollerance, double imu_rpy_weight)
{
  utils_ = utils;
  odometry_frame_ = odometry_frame;
  base_link_frame_ = base_link_frame;
  n_scan_ = n_scan;
  horizon_scan_ = horizon_scan;
  surrounding_key_frame_search_radius_ = surrounding_key_frame_search_radius;
  min_edge_feature_number_ = min_edge_feature_number;
  min_surface_feature_number_ = min_surface_feature_number;
  rotation_tollerance_ = rotation_tollerance;
  z_tollerance_ = z_tollerance;
  imu_rpy_weight_ = imu_rpy_weight;
  surrounding_key_frame_adding_angle_threshold_ = surrounding_key_frame_adding_angle_threshold;
  surrounding_key_frame_adding_dist_threshold_ = surrounding_key_frame_adding_dist_threshold;
  surrounding_key_frame_density_ = surrounding_key_frame_density;
  mapping_corner_leaf_size_ = mapping_corner_leaf_size;
  mapping_surface_leaf_size_ = mapping_surface_leaf_size;
  map_corner_path_ = corner_map_path;
  map_surface_path_ = surface_map_path;

  gtsam::ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.1;
  parameters.relinearizeSkip = 1;
  isam_ = new gtsam::ISAM2(parameters);

  allocate_memory(pub_map_corner, pub_map_surface, now);

};

void FeatureMatching::allocate_memory(
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_map_corner,
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_map_surface,
  rclcpp::Time now) {
  cloud_key_poses_3d_.reset(new pcl::PointCloud<PointType>());
  cloud_key_poses_6d_.reset(new pcl::PointCloud<PointTypePose>());
  copy_cloud_key_poses_3d_.reset(new pcl::PointCloud<PointType>());
  copy_cloud_key_poses_6d_.reset(new pcl::PointCloud<PointTypePose>());

  kdtree_surrounding_key_poses_.reset(new pcl::KdTreeFLANN<PointType>());
  kdtree_history_key_poses_.reset(new pcl::KdTreeFLANN<PointType>());

  laser_cloud_corner_last_.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
  laser_cloud_surface_last_.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
  laser_cloud_corner_last_ds_.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
  laser_cloud_surface_last_ds_.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

  laser_cloud_ori_.reset(new pcl::PointCloud<PointType>());
  coeff_sel_.reset(new pcl::PointCloud<PointType>());

  laser_cloud_ori_corner_vec_.resize(n_scan_ * horizon_scan_);
  coeff_sel_corner_vec_.resize(n_scan_ * horizon_scan_);
  laser_cloud_ori_corner_flag_.resize(n_scan_ * horizon_scan_);
  laser_cloud_ori_surface_vec_.resize(n_scan_ * horizon_scan_);
  coeff_sel_surface_vec_.resize(n_scan_ * horizon_scan_);
  laser_cloud_ori_surface_flag_.resize(n_scan_ * horizon_scan_);

  std::fill(laser_cloud_ori_corner_flag_.begin(), laser_cloud_ori_corner_flag_.end(), false);
  std::fill(laser_cloud_ori_surface_flag_.begin(), laser_cloud_ori_surface_flag_.end(), false);

  laser_cloud_corner_from_map_.reset(new pcl::PointCloud<PointType>());
  laser_cloud_surface_from_map_.reset(new pcl::PointCloud<PointType>());
  laser_cloud_corner_from_map_ds_.reset(new pcl::PointCloud<PointType>());
  laser_cloud_surface_from_map_ds_.reset(new pcl::PointCloud<PointType>());

  map_corner_octree_.reset(new pcl::octree::OctreePointCloudSearch<PointType>(0.2));
  map_surface_octree_.reset(new pcl::octree::OctreePointCloudSearch<PointType>(0.4));
  down_size_filter_corner_.setLeafSize(mapping_corner_leaf_size_, mapping_corner_leaf_size_, mapping_corner_leaf_size_);
  down_size_filter_surface_.setLeafSize(mapping_surface_leaf_size_, mapping_surface_leaf_size_, mapping_surface_leaf_size_);
  down_size_filter_icp_.setLeafSize(mapping_surface_leaf_size_, mapping_surface_leaf_size_, mapping_surface_leaf_size_);
  down_size_filter_surrounding_key_poses_.setLeafSize(
    surrounding_key_frame_density_, surrounding_key_frame_density_, surrounding_key_frame_density_); // for surrounding key poses of scan-to-map optimization


  kdtree_corner_from_map_.reset(new pcl::KdTreeFLANN<PointType>());
  kdtree_surface_from_map_.reset(new pcl::KdTreeFLANN<PointType>());

  for (int i = 0; i < 6; ++i){
    transform_to_be_mapped[i] = 0;
  }

  mat_p_.setZero();

  map_corner_.reset(new pcl::PointCloud<PointType>());
  if (pcl::io::loadPCDFile<PointType> (map_corner_path_, *map_corner_) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read corner cloud");
    PCL_ERROR ("Couldn't read corner cloud");
    PCL_ERROR ("Couldn't read corner cloud \n");
  }
  std::cout << "map_surface_path_: " << map_surface_path_ << std::endl;
  map_surface_.reset(new pcl::PointCloud<PointType>());
  if (pcl::io::loadPCDFile<PointType> (map_surface_path_, *map_surface_) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read surface cloud \n");
    PCL_ERROR ("Couldn't read surface cloud \n");
    PCL_ERROR ("Couldn't read surface cloud \n");
  }

  laser_cloud_corner_from_map_ds_num_ = map_corner_->size();
  laser_cloud_surface_from_map_ds_num_ = map_surface_->size();
//  laser_cloud_corner_from_map_ = map_corner_;
//  laser_cloud_corner_from_map_ds_ = map_corner_;
//  laser_cloud_surface_from_map_ = map_surface_;
//  laser_cloud_surface_from_map_ds_ = map_surface_;

  map_corner_clipped_.reset(new pcl::PointCloud<PointType>());
  map_surface_clipped_.reset(new pcl::PointCloud<PointType>());
  map_corner_octree_->setInputCloud(map_corner_);
  map_corner_octree_->addPointsFromInputCloud();
  map_surface_octree_->setInputCloud(map_surface_);
  map_surface_octree_->addPointsFromInputCloud();

  Eigen::Vector3f min_point(65464.3 - 200, 40904.08 - 200, 44.6 - 200);
  Eigen::Vector3f max_point(65464.3 + 200, 40904.08 + 200, 44.6 + 200);
//  -65464.3, -40904.08, -44.6

  std::vector<int> point_indices_in_box_corner;
  std::vector<int> point_indices_in_box_surface;
  map_corner_octree_->boxSearch(min_point, max_point, point_indices_in_box_corner);
  map_surface_octree_->boxSearch(min_point, max_point, point_indices_in_box_surface);

  //    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_region (new pcl::PointCloud<pcl::PointXYZ>);
  laser_cloud_corner_from_map_ds_.reset();
  laser_cloud_surface_from_map_ds_.reset();
  std::cout << "point_indices_in_box_corner: " << point_indices_in_box_corner.size() << std::endl;
  std::cout << "map_corner_: " << map_corner_->size() << std::endl;
  for (const auto& index : point_indices_in_box_corner) {
    laser_cloud_corner_from_map_ds_->push_back(map_corner_->points[index]);
  }
  for (const auto& index : point_indices_in_box_surface) {
    laser_cloud_surface_from_map_ds_->push_back(map_surface_->points[index]);
  }


  sensor_msgs::msg::PointCloud2 ros_cloud_corner;
  pcl::toROSMsg(*map_corner_, ros_cloud_corner);
  sensor_msgs::msg::PointCloud2 ros_cloud_surface;
  pcl::toROSMsg(*map_surface_, ros_cloud_surface);

  ros_cloud_corner.header.frame_id = "map";
  ros_cloud_corner.header.stamp = now;
  ros_cloud_surface.header.frame_id = "map";
  ros_cloud_surface.header.stamp = now;

  pub_map_corner->publish(ros_cloud_corner);
  pub_map_surface->publish(ros_cloud_surface);

//  RCLCPP_INFO(this->get_logger(), "\n\n\n\n\n\n\nPUBLISHED\n\n\n\n\n\n\n");
}

void FeatureMatching::laser_cloud_info_handler(
  const Utils::CloudInfo msg_in, std_msgs::msg::Header header,
  pcl::PointCloud<PointType>::Ptr cloud_corner, pcl::PointCloud<PointType>::Ptr cloud_surface,
  const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr & pub_odom_laser,
  const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr & pub_odom_laser_incremental,
  const std::unique_ptr<tf2_ros::TransformBroadcaster> & br,
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_key_poses,
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_recent_key_frames,
//  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_cloud_registered,
  const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr & pub_path)
{
  // extract time stamp
  time_laser_info_stamp_ = header.stamp;
  time_laser_info_cur_ = utils_->stamp2Sec(header.stamp);

  // extract info and feature cloud
  cloud_info_ = msg_in;
  laser_cloud_corner_last_ = cloud_corner;
  laser_cloud_surface_last_ = cloud_surface;
//  pcl::fromROSMsg(cloud_corner,  *laser_cloud_corner_last_);
//  pcl::fromROSMsg(cloud_surface, *laserCloudSurfLast);

//  std::lock_guard<std::mutex> lock(mtx);

  double mapping_process_interval = 0.1;

  static double timeLastProcessing = -1;
  if (time_laser_info_cur_ - timeLastProcessing >= mapping_process_interval)
  {
    timeLastProcessing = time_laser_info_cur_;

    std::cout << "-------------------------------------------------------------------" << std::endl;

    auto start_update_initial_guess = std::chrono::high_resolution_clock::now();
    update_initial_guess();
    auto end_update_initial_guess = std::chrono::high_resolution_clock::now();
    auto duration_update_initial_guess = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_update_initial_guess - start_update_initial_guess);
    std::cout << "end_update_initial_guess time: " << std::setprecision(6) << duration_update_initial_guess.count() << std::endl;

    auto start_extract_surrounding_key_frames = std::chrono::high_resolution_clock::now();
    extract_surrounding_key_frames();
    auto end_extract_surrounding_key_frames = std::chrono::high_resolution_clock::now();
    auto duration_extract_surrounding_key_frames = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_extract_surrounding_key_frames - start_extract_surrounding_key_frames);
    std::cout << "extract_surrounding_key_frames time: " << std::setprecision(6) << duration_extract_surrounding_key_frames.count() << std::endl;

    auto start_downsample_current_scan = std::chrono::high_resolution_clock::now();
    downsample_current_scan();
    auto end_downsample_current_scan = std::chrono::high_resolution_clock::now();
    auto duration_downsample_current_scan = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_downsample_current_scan - start_downsample_current_scan);
    std::cout << "downsample_current_scan time: " << std::setprecision(6) << duration_downsample_current_scan.count() << std::endl;

    auto start_scan_to_map_optimization = std::chrono::high_resolution_clock::now();
    scan_to_map_optimization();
    auto end_scan_to_map_optimization = std::chrono::high_resolution_clock::now();
    auto duration_scan_to_map_optimization = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_scan_to_map_optimization - start_scan_to_map_optimization);
    std::cout << "scan_to_map_optimization time: " << std::setprecision(6) << duration_scan_to_map_optimization.count() << std::endl;


    auto start_save_key_frames_and_factor = std::chrono::high_resolution_clock::now();
    save_key_frames_and_factor();
    auto end_save_key_frames_and_factor = std::chrono::high_resolution_clock::now();
    auto duration_save_key_frames_and_factor = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_save_key_frames_and_factor - start_save_key_frames_and_factor);
    std::cout << "save_key_frames_and_factor time: " << std::setprecision(6) << duration_save_key_frames_and_factor.count() << std::endl;

    auto start_correct_poses = std::chrono::high_resolution_clock::now();
    correct_poses();
    auto end_correct_poses = std::chrono::high_resolution_clock::now();
    auto duration_correct_poses = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_correct_poses - start_correct_poses);
    std::cout << "correct_poses time: " << std::setprecision(6) << duration_correct_poses.count() << std::endl;

    auto start_publish_odometry = std::chrono::high_resolution_clock::now();
    publish_odometry(pub_odom_laser, pub_odom_laser_incremental, br);
    auto end_publish_odometry = std::chrono::high_resolution_clock::now();
    auto duration_publish_odometry = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_publish_odometry - start_publish_odometry);
    std::cout << "publish_odometry time: " << std::setprecision(6) << duration_publish_odometry.count() << std::endl;

    auto start_publish_frames = std::chrono::high_resolution_clock::now();
    publish_frames(pub_key_poses, pub_recent_key_frames, pub_path);
    auto end_publish_frames = std::chrono::high_resolution_clock::now();
    auto duration_publish_frames = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_publish_frames - start_publish_frames);
    std::cout << "publish_frames time: " << std::setprecision(6) << duration_publish_frames.count() << std::endl;
  }

  std::cout << "-------------------------------------------------------------------\n\n" << std::endl;
}

void FeatureMatching::point_associate_to_map(PointType const * const pi, PointType * const po)
{
  po->x = trans_point_associate_to_map_(0,0) * pi->x + trans_point_associate_to_map_(0,1) * pi->y + trans_point_associate_to_map_(0,2) * pi->z + trans_point_associate_to_map_(0,3);
  po->y = trans_point_associate_to_map_(1,0) * pi->x + trans_point_associate_to_map_(1,1) * pi->y + trans_point_associate_to_map_(1,2) * pi->z + trans_point_associate_to_map_(1,3);
  po->z = trans_point_associate_to_map_(2,0) * pi->x + trans_point_associate_to_map_(2,1) * pi->y + trans_point_associate_to_map_(2,2) * pi->z + trans_point_associate_to_map_(2,3);
  po->intensity = pi->intensity;
}

pcl::PointCloud<PointType>::Ptr FeatureMatching::transform_point_cloud(
  pcl::PointCloud<PointType>::Ptr cloud_in, PointTypePose* transform_in)
{
  pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

  int cloudSize = cloud_in->size();
  cloudOut->resize(cloudSize);

  Eigen::Affine3f transCur = pcl::getTransformation(
    transform_in->x, transform_in->y, transform_in->z,
    transform_in->roll, transform_in->pitch, transform_in->yaw);

  //        #pragma omp parallel for num_threads(numberOfCores)
  for (int i = 0; i < cloudSize; ++i)
  {
    const auto &pointFrom = cloud_in->points[i];
    cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
    cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
    cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
    cloudOut->points[i].intensity = pointFrom.intensity;
  }
  return cloudOut;
}

gtsam::Pose3 FeatureMatching::pcl_point_to_gtsam_pose3(PointTypePose this_point)
{
  return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(this_point.roll), double(this_point.pitch), double(this_point.yaw)),
                      gtsam::Point3(double(this_point.x),    double(this_point.y),     double(this_point.z)));

}

gtsam::Pose3 FeatureMatching::trans_to_gtsam_pose(float transform_in[])
{
  return gtsam::Pose3(gtsam::Rot3::RzRyRx(transform_in[0], transform_in[1], transform_in[2]),
                      gtsam::Point3(transform_in[3], transform_in[4], transform_in[5]));
}

Eigen::Affine3f FeatureMatching::pcl_point_to_affine3f(PointTypePose this_point)
{
   return pcl::getTransformation(this_point.x, this_point.y, this_point.z,
                                this_point.roll, this_point.pitch, this_point.yaw);
}

Eigen::Affine3f FeatureMatching::trans_to_affine3f(float transform_in[])
{
  return pcl::getTransformation(transform_in[3], transform_in[4], transform_in[5], transform_in[0], transform_in[1], transform_in[2]);
}

PointTypePose FeatureMatching::trans_to_point_type_pose(float transform_in[])
{
  PointTypePose thisPose6D;
  thisPose6D.x = transform_in[3];
  thisPose6D.y = transform_in[4];
  thisPose6D.z = transform_in[5];
  thisPose6D.roll  = transform_in[0];
  thisPose6D.pitch = transform_in[1];
  thisPose6D.yaw   = transform_in[2];
  return thisPose6D;
}

void FeatureMatching::update_initial_guess() {
  // save current transformation before any processing
  incremental_odometry_affine_front_= trans_to_affine3f(transform_to_be_mapped);

  static Eigen::Affine3f lastImuTransformation;
  // initialization

  if (cloud_key_poses_3d_->points.empty())
  {
    transform_to_be_mapped[0] = cloud_info_.imu_roll_init;
    transform_to_be_mapped[1] = cloud_info_.imu_pitch_init;
    transform_to_be_mapped[2] = cloud_info_.imu_yaw_init;

//    if (!useImuHeadingInitialization)
//      transformTobeMapped[2] = 0;

    // TODO: MAKE TOPIC HERE
//    lastImuTransformation = pcl::getTransformation(-66458, -43619, -42, cloud_info_.imu_roll_init, cloud_info_.imu_pitch_init, cloud_info_.imu_yaw_init); // save imu before return;
    lastImuTransformation = pcl::getTransformation(-65464.3, -40904.08, -44.6, cloud_info_.imu_roll_init, cloud_info_.imu_pitch_init, cloud_info_.imu_yaw_init); // save imu before return;
//    lastImuTransformation = pcl::getTransformation(0, 0, 0, cloud_info_.imu_roll_init, cloud_info_.imu_pitch_init, cloud_info_.imu_yaw_init); // save imu before return;
    return;
  }

  // use imu pre-integration estimation for pose guess
  static bool lastImuPreTransAvailable = false;
  static Eigen::Affine3f lastImuPreTransformation;
  if (cloud_info_.odom_available == true)
  {
    Eigen::Affine3f transBack = pcl::getTransformation(
      cloud_info_.initial_guess_x, cloud_info_.initial_guess_y, cloud_info_.initial_guess_z,
      cloud_info_.initial_guess_roll, cloud_info_.initial_guess_pitch, cloud_info_.initial_guess_yaw);
    if (lastImuPreTransAvailable == false)
    {
      lastImuPreTransformation = transBack;
      lastImuPreTransAvailable = true;
    } else {
      Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
      Eigen::Affine3f transTobe = trans_to_affine3f(transform_to_be_mapped);
      Eigen::Affine3f transFinal = transTobe * transIncre;
      pcl::getTranslationAndEulerAngles(transFinal, transform_to_be_mapped[3], transform_to_be_mapped[4], transform_to_be_mapped[5],
                                        transform_to_be_mapped[0], transform_to_be_mapped[1], transform_to_be_mapped[2]);

      lastImuPreTransformation = transBack;

      lastImuTransformation = pcl::getTransformation(0, 0, 0, cloud_info_.imu_roll_init, cloud_info_.imu_pitch_init, cloud_info_.imu_yaw_init); // save imu before return;
      return;
    }
  }

  // use imu incremental estimation for pose guess (only rotation)
  if (cloud_info_.imu_available == true)
  {
    Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloud_info_.imu_roll_init, cloud_info_.imu_pitch_init, cloud_info_.imu_yaw_init);
    Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

    Eigen::Affine3f transTobe = trans_to_affine3f(transform_to_be_mapped);
    Eigen::Affine3f transFinal = transTobe * transIncre;
    pcl::getTranslationAndEulerAngles(transFinal, transform_to_be_mapped[3], transform_to_be_mapped[4], transform_to_be_mapped[5],
                                      transform_to_be_mapped[0], transform_to_be_mapped[1], transform_to_be_mapped[2]);

    lastImuTransformation = pcl::getTransformation(0, 0, 0, cloud_info_.imu_roll_init, cloud_info_.imu_pitch_init, cloud_info_.imu_yaw_init); // save imu before return;
    return;
  }
}

void FeatureMatching::extract_nearby()
{
  pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
  std::vector<int> pointSearchInd;
  std::vector<float> pointSearchSqDis;

  // extract all the nearby key poses and downsample them
  kdtree_surrounding_key_poses_->setInputCloud(cloud_key_poses_3d_); // create kd-tree
  kdtree_surrounding_key_poses_->radiusSearch(cloud_key_poses_3d_->back(), (double)surrounding_key_frame_search_radius_, pointSearchInd, pointSearchSqDis);
  for (int i = 0; i < (int)pointSearchInd.size(); ++i)
  {
    int id = pointSearchInd[i];
    surroundingKeyPoses->push_back(cloud_key_poses_3d_->points[id]);
  }

  down_size_filter_surrounding_key_poses_.setInputCloud(surroundingKeyPoses);
  down_size_filter_surrounding_key_poses_.filter(*surroundingKeyPosesDS);
  for(auto& pt : surroundingKeyPosesDS->points)
  {
    kdtree_surrounding_key_poses_->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
    pt.intensity = cloud_key_poses_3d_->points[pointSearchInd[0]].intensity;
  }

  // also extract some latest key frames in case the robot rotates in one position
  int numPoses = cloud_key_poses_3d_ ->size();
  for (int i = numPoses-1; i >= 0; --i)
  {
    if (time_laser_info_cur_ - cloud_key_poses_6d_->points[i].time < 10.0)
      surroundingKeyPosesDS->push_back(cloud_key_poses_3d_->points[i]);
    else
      break;
  }

  extract_cloud();
}

void FeatureMatching::extract_cloud()
{

  float position_difference_from_last_map_load_point =
    std::sqrt(std::pow(transform_to_be_mapped[0] - transform_to_be_mapped_old_[0], 2) +
              std::pow(transform_to_be_mapped[1] - transform_to_be_mapped_old_[1], 2) +
              std::pow(transform_to_be_mapped[2] - transform_to_be_mapped_old_[2], 2));

  if (position_difference_from_last_map_load_point >= 100){
    Eigen::Vector3f min_point(transform_to_be_mapped[0] - 200, transform_to_be_mapped[1] - 200, transform_to_be_mapped[2] - 200);
    Eigen::Vector3f max_point(transform_to_be_mapped[0] + 200, transform_to_be_mapped[1] + 200, transform_to_be_mapped[2] + 200);

//    PointType min_point, max_point;
//    min_point.x = transform_to_be_mapped[0] - 200;
//    min_point.y = transform_to_be_mapped[1] - 200;
//    min_point.z = transform_to_be_mapped[2] - 200;
//    min_point.intensity = 255.0f;
//    max_point.x = transform_to_be_mapped[0] + 200;
//    max_point.y = transform_to_be_mapped[1] + 200;
//    max_point.z = transform_to_be_mapped[2] + 200;
//    max_point.intensity = 255.0f;

    std::vector<int> point_indices_in_box_corner;
    std::vector<int> point_indices_in_box_surface;
    map_corner_octree_->boxSearch(min_point, max_point, point_indices_in_box_corner);
    map_surface_octree_->boxSearch(min_point, max_point, point_indices_in_box_surface);

//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_region (new pcl::PointCloud<pcl::PointXYZ>);
    laser_cloud_corner_from_map_ds_.reset();
    laser_cloud_surface_from_map_ds_.reset();
    for (const auto& index : point_indices_in_box_corner) {
      laser_cloud_corner_from_map_ds_->points.push_back(map_corner_->points[index]);
    }
    for (const auto& index : point_indices_in_box_surface) {
      laser_cloud_surface_from_map_ds_->points.push_back(map_surface_->points[index]);
    }
  } else {
    return;
  }
  if (laser_cloud_map_container_.size() > 1000)
     laser_cloud_map_container_.clear();
}

void FeatureMatching::extract_surrounding_key_frames()
{
  if (cloud_key_poses_3d_->points.empty() == true)
    return;

  // if (loopClosureEnableFlag == true)
  // {
  //     extractForLoopClosure();
  // } else {
  //     extractNearby();
  // }

  extract_nearby();
}

void FeatureMatching::downsample_current_scan()
{
    // Downsample cloud from current scan
   laser_cloud_corner_last_ds_->clear();
   down_size_filter_corner_.setInputCloud(laser_cloud_corner_last_);
   down_size_filter_corner_.filter(*laser_cloud_corner_last_ds_);
   laser_cloud_corner_last_ds_num_ = laser_cloud_corner_last_ds_->size();

   laser_cloud_surface_last_ds_->clear();
   down_size_filter_surface_.setInputCloud(laser_cloud_surface_last_);
   down_size_filter_surface_.filter(*laser_cloud_surface_last_ds_);
   laser_cloud_surface_last_ds_num_ = laser_cloud_surface_last_ds_->size();
}

void FeatureMatching::update_point_associate_to_map()
{
  trans_point_associate_to_map_ = trans_to_affine3f(transform_to_be_mapped);
}

void FeatureMatching::corner_optimization()
{
  update_point_associate_to_map();

  //        #pragma omp parallel for num_threads(numberOfCores)
  for (int i = 0; i < laser_cloud_corner_last_ds_num_; i++)
  {
    PointType pointOri, pointSel, coeff;
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    pointOri = laser_cloud_corner_last_ds_->points[i];
    point_associate_to_map(&pointOri, &pointSel);
    kdtree_corner_from_map_->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

    cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

    if (pointSearchSqDis[4] < 1.0) {
      float cx = 0, cy = 0, cz = 0;
      for (int j = 0; j < 5; j++) {
        cx += laser_cloud_corner_from_map_ds_->points[pointSearchInd[j]].x;
        cy += laser_cloud_corner_from_map_ds_->points[pointSearchInd[j]].y;
        cz += laser_cloud_corner_from_map_ds_->points[pointSearchInd[j]].z;
      }
      cx /= 5; cy /= 5;  cz /= 5;

      float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
      for (int j = 0; j < 5; j++) {
        float ax = laser_cloud_corner_from_map_ds_->points[pointSearchInd[j]].x - cx;
        float ay = laser_cloud_corner_from_map_ds_->points[pointSearchInd[j]].y - cy;
        float az = laser_cloud_corner_from_map_ds_->points[pointSearchInd[j]].z - cz;

        a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
        a22 += ay * ay; a23 += ay * az;
        a33 += az * az;
      }
      a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

      matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
      matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
      matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

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

        float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                          + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                          + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

        float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

        float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                    + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

        float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                     - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

        float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                     + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

        float ld2 = a012 / l12;

        float s = 1 - 0.9 * fabs(ld2);

        coeff.x = s * la;
        coeff.y = s * lb;
        coeff.z = s * lc;
        coeff.intensity = s * ld2;

        if (s > 0.1) {
          laser_cloud_ori_corner_vec_[i] = pointOri;
          coeff_sel_corner_vec_[i] = coeff;
          laser_cloud_ori_corner_flag_[i] = true;
        }
      }
    }
  }
}

void FeatureMatching::surface_optimization()
{
  update_point_associate_to_map();

//  #pragma omp parallel for num_threads(10)
  for (int i = 0; i < laser_cloud_surface_last_ds_num_; i++)
  {

    PointType pointOri, pointSel, coeff;
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    pointOri = laser_cloud_surface_last_ds_->points[i];
    point_associate_to_map(&pointOri, &pointSel);

//    auto start = std::chrono::high_resolution_clock::now();
    kdtree_surface_from_map_->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
//    auto end = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
//      end - start);
//    std::cout << "\t\tkdtree_surface_from_map_->nearestKSearch time: " << std::setprecision(10) << duration.count() << std::endl;

    Eigen::Matrix<float, 5, 3> matA0;
    Eigen::Matrix<float, 5, 1> matB0;
    Eigen::Vector3f matX0;

    matA0.setZero();
    matB0.fill(-1);
    matX0.setZero();


    if (pointSearchSqDis[4] < 1.0) {
      for (int j = 0; j < 5; j++) {
        matA0(j, 0) = laser_cloud_surface_from_map_ds_->points[pointSearchInd[j]].x;
        matA0(j, 1) = laser_cloud_surface_from_map_ds_->points[pointSearchInd[j]].y;
        matA0(j, 2) = laser_cloud_surface_from_map_ds_->points[pointSearchInd[j]].z;
      }

      matX0 = matA0.colPivHouseholderQr().solve(matB0);

      float pa = matX0(0, 0);
      float pb = matX0(1, 0);
      float pc = matX0(2, 0);
      float pd = 1;

      float ps = sqrt(pa * pa + pb * pb + pc * pc);
      pa /= ps; pb /= ps; pc /= ps; pd /= ps;

      bool planeValid = true;
      for (int j = 0; j < 5; j++) {
        if (fabs(pa * laser_cloud_surface_from_map_ds_->points[pointSearchInd[j]].x +
                 pb * laser_cloud_surface_from_map_ds_->points[pointSearchInd[j]].y +
                 pc * laser_cloud_surface_from_map_ds_->points[pointSearchInd[j]].z + pd) > 0.2) {
          planeValid = false;
          break;
        }
      }

      if (planeValid) {
        float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

        float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                                                  + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

        coeff.x = s * pa;
        coeff.y = s * pb;
        coeff.z = s * pc;
        coeff.intensity = s * pd2;

        if (s > 0.1) {
          laser_cloud_ori_surface_vec_[i] = pointOri;
          coeff_sel_surface_vec_[i] = coeff;
          laser_cloud_ori_surface_flag_[i] = true;
        }
      }
    }
  }
}

void FeatureMatching::combine_optimization_coeffs()
{
  // combine corner coeffs
  for (int i = 0; i < laser_cloud_corner_last_ds_num_; ++i){
    if (laser_cloud_ori_corner_flag_[i] == true){
      laser_cloud_ori_->push_back(laser_cloud_ori_corner_vec_[i]);
      coeff_sel_->push_back(coeff_sel_corner_vec_[i]);
    }
  }
  // combine surf coeffs
  for (int i = 0; i < laser_cloud_surface_last_ds_num_; ++i){
    if (laser_cloud_ori_surface_flag_[i] == true){
      laser_cloud_ori_->push_back(laser_cloud_ori_surface_vec_[i]);
      coeff_sel_->push_back(coeff_sel_surface_vec_[i]);
    }
  }
  // reset flag for next iteration
  std::fill(laser_cloud_ori_corner_flag_.begin(), laser_cloud_ori_corner_flag_.end(), false);
  std::fill(laser_cloud_ori_surface_flag_.begin(), laser_cloud_ori_surface_flag_.end(), false);
}

bool FeatureMatching::lm_optimization(int iter_count)
{
  // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
  // lidar <- camera      ---     camera <- lidar
  // x = z                ---     x = y
  // y = x                ---     y = z
  // z = y                ---     z = x
  // roll = yaw           ---     roll = pitch
  // pitch = roll         ---     pitch = yaw
  // yaw = pitch          ---     yaw = roll

  // lidar -> camera
  float srx = sin(transform_to_be_mapped[1]);
  float crx = cos(transform_to_be_mapped[1]);
  float sry = sin(transform_to_be_mapped[2]);
  float cry = cos(transform_to_be_mapped[2]);
  float srz = sin(transform_to_be_mapped[0]);
  float crz = cos(transform_to_be_mapped[0]);

  int laserCloudSelNum = laser_cloud_ori_->size();
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
    pointOri.x = laser_cloud_ori_->points[i].y;
    pointOri.y = laser_cloud_ori_->points[i].z;
    pointOri.z = laser_cloud_ori_->points[i].x;
    // lidar -> camera
    coeff.x = coeff_sel_->points[i].y;
    coeff.y = coeff_sel_->points[i].z;
    coeff.z = coeff_sel_->points[i].x;
    coeff.intensity = coeff_sel_->points[i].intensity;
    // in camera
    float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

    float ary = ((cry*srx*srz - crz*sry)*pointOri.x
                 + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                + ((-cry*crz - srx*sry*srz)*pointOri.x
                   + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

    float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
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

  if (iter_count == 0) {

    cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

    cv::eigen(matAtA, matE, matV);
    matV.copyTo(matV2);

    is_degenerate_ = false;
    float eignThre[6] = {100, 100, 100, 100, 100, 100};
    for (int i = 5; i >= 0; i--) {
      if (matE.at<float>(0, i) < eignThre[i]) {
        for (int j = 0; j < 6; j++) {
          matV2.at<float>(i, j) = 0;
        }
        is_degenerate_ = true;
      } else {
        break;
      }
    }
    matP = matV.inv() * matV2;
  }

  if (is_degenerate_)
  {
    cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
    matX.copyTo(matX2);
    matX = matP * matX2;
  }

  transform_to_be_mapped[0] += matX.at<float>(0, 0);
  transform_to_be_mapped[1] += matX.at<float>(1, 0);
  transform_to_be_mapped[2] += matX.at<float>(2, 0);
  transform_to_be_mapped[3] += matX.at<float>(3, 0);
  transform_to_be_mapped[4] += matX.at<float>(4, 0);
  transform_to_be_mapped[5] += matX.at<float>(5, 0);

  float deltaR = sqrt(
    std::pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
    std::pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
    std::pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
  float deltaT = sqrt(
    std::pow(matX.at<float>(3, 0) * 100, 2) +
    std::pow(matX.at<float>(4, 0) * 100, 2) +
    std::pow(matX.at<float>(5, 0) * 100, 2));

  if (deltaR < 0.05 && deltaT < 0.05) {
    return true; // converged
  }
  return false; // keep optimizing
}


void FeatureMatching::scan_to_map_optimization()
{
  if (cloud_key_poses_3d_->points.empty())
    return;

  if (laser_cloud_corner_last_ds_num_ > min_edge_feature_number_ && laser_cloud_surface_last_ds_num_ > min_surface_feature_number_)
  {
    kdtree_corner_from_map_->setInputCloud(laser_cloud_corner_from_map_ds_);
    kdtree_surface_from_map_->setInputCloud(laser_cloud_surface_from_map_ds_);

    for (int iterCount = 0; iterCount < 30; iterCount++)
    {
      laser_cloud_ori_->clear();
      coeff_sel_->clear();

      auto start_corner_optimization = std::chrono::high_resolution_clock::now();
      corner_optimization();
      auto end_corner_optimization = std::chrono::high_resolution_clock::now();
      auto duration_corner_optimization = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_corner_optimization - start_corner_optimization);
      std::cout << "\tcorner_optimization time: " << std::setprecision(6) << duration_corner_optimization.count() << std::endl;

      auto start_surface_optimization = std::chrono::high_resolution_clock::now();
      surface_optimization();
      auto end_surface_optimization = std::chrono::high_resolution_clock::now();
      auto duration_surface_optimization = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_surface_optimization - start_surface_optimization);
      std::cout << "\tsurface_optimization time: " << std::setprecision(6) << duration_surface_optimization.count() << std::endl;

      auto start_combine_optimization_coeffs = std::chrono::high_resolution_clock::now();
      combine_optimization_coeffs();
      auto end_combine_optimization_coeffs = std::chrono::high_resolution_clock::now();
      auto duration_combine_optimization_coeffs = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_combine_optimization_coeffs - start_combine_optimization_coeffs);
      std::cout << "\tcombine_optimization_coeffs time: " << std::setprecision(6) << duration_combine_optimization_coeffs.count() << std::endl;


      if (lm_optimization(iterCount) == true)
        break;
    }

    transform_update();
  } else {
//    RCLCPP_WARN(get_logger(), "Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
    std::cout << "Not enough features! Only " << laser_cloud_corner_last_ds_num_ << " edge and " << laser_cloud_surface_last_ds_num_ << "planar features available" << std::endl;
  }
}

void FeatureMatching::transform_update()
{
  if (cloud_info_.imu_available == true)
  {
    if (std::abs(cloud_info_.imu_pitch_init) < 1.4)
    {
      double imuWeight = imu_rpy_weight_;
      tf2::Quaternion imuQuaternion;
      tf2::Quaternion transformQuaternion;
      double rollMid, pitchMid, yawMid;

      // slerp roll
      transformQuaternion.setRPY(transform_to_be_mapped[0], 0, 0);
      imuQuaternion.setRPY(cloud_info_.imu_roll_init, 0, 0);
      tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
      transform_to_be_mapped[0] = rollMid;

      // slerp pitch
      transformQuaternion.setRPY(0, transform_to_be_mapped[1], 0);
      imuQuaternion.setRPY(0, cloud_info_.imu_pitch_init, 0);
      tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
      transform_to_be_mapped[1] = pitchMid;
    }
  }

  transform_to_be_mapped[0] = constraint_transformation(transform_to_be_mapped[0], rotation_tollerance_);
  transform_to_be_mapped[1] = constraint_transformation(transform_to_be_mapped[1], rotation_tollerance_);
  transform_to_be_mapped[5] = constraint_transformation(transform_to_be_mapped[5], z_tollerance_);

  incremental_odometry_affine_back_ = trans_to_affine3f(transform_to_be_mapped);
}

float FeatureMatching::constraint_transformation(float value, float limit)
{
  if (value < -limit)
    value = -limit;
  if (value > limit)
    value = limit;

  return value;
}

bool FeatureMatching::save_frame() {
  if (cloud_key_poses_3d_->points.empty())
    return true;

//  if (sensor == SensorType::LIVOX)
//  {
//    if (timeLaserInfoCur - cloudKeyPoses6D->back().time > 1.0)
//      return true;
//  }

  Eigen::Affine3f transStart = pcl_point_to_affine3f(cloud_key_poses_6d_->back());
  Eigen::Affine3f transFinal = pcl::getTransformation(transform_to_be_mapped[3], transform_to_be_mapped[4], transform_to_be_mapped[5],
                                                      transform_to_be_mapped[0], transform_to_be_mapped[1], transform_to_be_mapped[2]);
  Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
  float x, y, z, roll, pitch, yaw;
  pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

  if (abs(roll)  < surrounding_key_frame_adding_angle_threshold_ &&
      abs(pitch) < surrounding_key_frame_adding_angle_threshold_ &&
      abs(yaw)   < surrounding_key_frame_adding_angle_threshold_ &&
      sqrt(x*x + y*y + z*z) < surrounding_key_frame_adding_dist_threshold_)
    return false;

  return true;
}

void FeatureMatching::add_odom_factor() {

  if (cloud_key_poses_3d_->points.empty())
  {
    gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
    gtsam_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, trans_to_gtsam_pose(transform_to_be_mapped), priorNoise));
    initial_estimate_.insert(0, trans_to_gtsam_pose(transform_to_be_mapped));
  }else{
    gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    gtsam::Pose3 poseFrom = pcl_point_to_gtsam_pose3(cloud_key_poses_6d_->points.back());
    gtsam::Pose3 poseTo   = trans_to_gtsam_pose(transform_to_be_mapped);
    gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(cloud_key_poses_3d_->size()-1, cloud_key_poses_3d_->size(), poseFrom.between(poseTo), odometryNoise));
    initial_estimate_.insert(cloud_key_poses_3d_->size(), poseTo);
  }
}

void FeatureMatching::add_gps_factor()
{
  if (gps_queue_.empty())
    return;

  // wait for system initialized and settles down
  if (cloud_key_poses_3d_->points.empty())
    return;
  else
  {
    if (utils_->pointDistance(cloud_key_poses_3d_->front(), cloud_key_poses_3d_->back()) < 5.0)
      return;
  }

  // pose covariance small, no need to correct
//  if (pose_covariance_(3,3) < poseCovThreshold && pose_covariance_(4,4) < poseCovThreshold)

  // instead of taking pose cov threshold as a parameter, directly use it as 0.2 meters
  if (pose_covariance_(3,3) < 0.04 && pose_covariance_(4,4) < 0.04)
    return;

  // last gps position
  static PointType lastGPSPoint;

  while (!gps_queue_.empty())
  {
    if (utils_->stamp2Sec(gps_queue_.front().header.stamp) < time_laser_info_cur_ - 0.2)
    {
      // message too old
      gps_queue_.pop_front();
    }
    else if (utils_->stamp2Sec(gps_queue_.front().header.stamp) > time_laser_info_cur_ + 0.2)
    {
      // message too new
      break;
    }
    else
    {
      nav_msgs::msg::Odometry thisGPS = gps_queue_.front();
      gps_queue_.pop_front();

      // GPS too noisy, skip
      float noise_x = thisGPS.pose.covariance[0];
      float noise_y = thisGPS.pose.covariance[7];
      float noise_z = thisGPS.pose.covariance[14];
      if (noise_x > 0.04 || noise_y > 0.04)
        continue;
      float gps_x = thisGPS.pose.pose.position.x;
      float gps_y = thisGPS.pose.pose.position.y;
      float gps_z = thisGPS.pose.pose.position.z;
//      if (!useGpsElevation)
//      {
        gps_z = transform_to_be_mapped[5];
        noise_z = 0.01;
//      }

      // GPS not properly initialized (0,0,0)
      if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
        continue;

      // Add GPS every a few meters
      PointType curGPSPoint;
      curGPSPoint.x = gps_x;
      curGPSPoint.y = gps_y;
      curGPSPoint.z = gps_z;
      if (utils_->pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
        continue;
      else
        lastGPSPoint = curGPSPoint;

      gtsam::Vector Vector3(3);
      Vector3 << std::max(noise_x, 1.0f), std::max(noise_y, 1.0f), std::max(noise_z, 1.0f);
      gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);
      gtsam::GPSFactor gps_factor(cloud_key_poses_3d_->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
      gtsam_graph_.add(gps_factor);

      a_loop_is_closed_ = true;
      break;
    }
  }
}


void FeatureMatching::add_loop_factor()
{
  if (loop_index_queue_.empty())
    return;

  for (int i = 0; i < (int)loop_index_queue_.size(); ++i)
  {
    int indexFrom = loop_index_queue_[i].first;
    int indexTo = loop_index_queue_[i].second;
    gtsam::Pose3 poseBetween = loop_pose_queue[i];
    gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loop_noise_queue_[i];
    gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
  }

  loop_index_queue_.clear();
  loop_pose_queue.clear();
  loop_noise_queue_.clear();
  a_loop_is_closed_ = true;
}


void FeatureMatching::save_key_frames_and_factor()
{
  if (save_frame() == false)
    return;

  // odom factor
  add_odom_factor();

  // gps factor
//  add_gps_factor();

  // loop factor
//  add_loop_factor();

  // cout << "****************************************************" << endl;
  // gtSAMgraph.print("GTSAM Graph:\n");

  // update iSAM
  isam_->update(gtsam_graph_, initial_estimate_);
  isam_->update();

  if (a_loop_is_closed_ == true)
  {
    isam_->update();
    isam_->update();
    isam_->update();
    isam_->update();
    isam_->update();
  }

  gtsam_graph_.resize(0);
  initial_estimate_.clear();

  //save key poses
  PointType thisPose3D;
  PointTypePose thisPose6D;
  gtsam::Pose3 latestEstimate;

  isam_current_estimate_ = isam_->calculateEstimate();
  latestEstimate = isam_current_estimate_.at<gtsam::Pose3>(isam_current_estimate_.size()-1);
  // cout << "****************************************************" << endl;
  // isamCurrentEstimate.print("Current estimate: ");

  thisPose3D.x = latestEstimate.translation().x();
  thisPose3D.y = latestEstimate.translation().y();
  thisPose3D.z = latestEstimate.translation().z();
  thisPose3D.intensity = cloud_key_poses_3d_->size(); // this can be used as index
  cloud_key_poses_3d_->push_back(thisPose3D);

  thisPose6D.x = thisPose3D.x;
  thisPose6D.y = thisPose3D.y;
  thisPose6D.z = thisPose3D.z;
  thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
  thisPose6D.roll  = latestEstimate.rotation().roll();
  thisPose6D.pitch = latestEstimate.rotation().pitch();
  thisPose6D.yaw   = latestEstimate.rotation().yaw();
  thisPose6D.time = time_laser_info_cur_;
  cloud_key_poses_6d_->push_back(thisPose6D);

  // cout << "****************************************************" << endl;
  // cout << "Pose covariance:" << endl;
  // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
  pose_covariance_ = isam_->marginalCovariance(isam_current_estimate_.size()-1);

  // save updated transform
  transform_to_be_mapped[0] = latestEstimate.rotation().roll();
  transform_to_be_mapped[1] = latestEstimate.rotation().pitch();
  transform_to_be_mapped[2] = latestEstimate.rotation().yaw();
  transform_to_be_mapped[3] = latestEstimate.translation().x();
  transform_to_be_mapped[4] = latestEstimate.translation().y();
  transform_to_be_mapped[5] = latestEstimate.translation().z();

  // save all the received edge and surf points
  pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
  pcl::copyPointCloud(*laser_cloud_corner_last_ds_,  *thisCornerKeyFrame);
  pcl::copyPointCloud(*laser_cloud_surface_last_ds_,    *thisSurfKeyFrame);

  // save key frame cloud
  corner_cloud_key_frames.push_back(thisCornerKeyFrame);
  surface_cloud_key_frames.push_back(thisSurfKeyFrame);

  // save path for visualization
  update_path(thisPose6D);
}


void FeatureMatching::correct_poses()
{
  if (cloud_key_poses_3d_->points.empty())
    return;

  if (a_loop_is_closed_ == true)
  {
    // clear map cache
    laser_cloud_map_container_.clear();
    // clear path
    global_path_.poses.clear();
    // update key poses
    int numPoses = isam_current_estimate_.size();
    for (int i = 0; i < numPoses; ++i)
    {
      cloud_key_poses_3d_->points[i].x = isam_current_estimate_.at<gtsam::Pose3>(i).translation().x();
      cloud_key_poses_3d_->points[i].y = isam_current_estimate_.at<gtsam::Pose3>(i).translation().y();
      cloud_key_poses_3d_->points[i].z = isam_current_estimate_.at<gtsam::Pose3>(i).translation().z();

      cloud_key_poses_6d_->points[i].x = cloud_key_poses_3d_->points[i].x;
      cloud_key_poses_6d_->points[i].y = cloud_key_poses_3d_->points[i].y;
      cloud_key_poses_6d_->points[i].z = cloud_key_poses_3d_->points[i].z;
      cloud_key_poses_6d_->points[i].roll  = isam_current_estimate_.at<gtsam::Pose3>(i).rotation().roll();
      cloud_key_poses_6d_->points[i].pitch = isam_current_estimate_.at<gtsam::Pose3>(i).rotation().pitch();
      cloud_key_poses_6d_->points[i].yaw   = isam_current_estimate_.at<gtsam::Pose3>(i).rotation().yaw();

      update_path(cloud_key_poses_6d_->points[i]);
    }

    a_loop_is_closed_ = false;
  }
}


void FeatureMatching::update_path(const PointTypePose& pose_in)
{
  geometry_msgs::msg::PoseStamped pose_stamped;
  pose_stamped.header.stamp = rclcpp::Time(pose_in.time * 1e9);
  pose_stamped.header.frame_id = odometry_frame_;
  pose_stamped.pose.position.x = pose_in.x;
  pose_stamped.pose.position.y = pose_in.y;
  pose_stamped.pose.position.z = pose_in.z;
  tf2::Quaternion q;
  q.setRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
  pose_stamped.pose.orientation.x = q.x();
  pose_stamped.pose.orientation.y = q.y();
  pose_stamped.pose.orientation.z = q.z();
  pose_stamped.pose.orientation.w = q.w();

  global_path_.poses.push_back(pose_stamped);
}


void FeatureMatching::publish_odometry(
  const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr & pub_odom_laser,
  const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr & pub_odom_laser_incremental,
  const std::unique_ptr<tf2_ros::TransformBroadcaster> & br)
{
  // Publish odometry for ROS (global)
  nav_msgs::msg::Odometry laserOdometryROS;
  laserOdometryROS.header.stamp = time_laser_info_stamp_;
  laserOdometryROS.header.frame_id = odometry_frame_;
  laserOdometryROS.child_frame_id = base_link_frame_;
  laserOdometryROS.pose.pose.position.x = transform_to_be_mapped[3];
  laserOdometryROS.pose.pose.position.y = transform_to_be_mapped[4];
  laserOdometryROS.pose.pose.position.z = transform_to_be_mapped[5];
  tf2::Quaternion quat_tf;
  quat_tf.setRPY(transform_to_be_mapped[0], transform_to_be_mapped[1], transform_to_be_mapped[2]);
  geometry_msgs::msg::Quaternion quat_msg;
  tf2::convert(quat_tf, quat_msg);
  laserOdometryROS.pose.pose.orientation = quat_msg;
  pub_odom_laser->publish(laserOdometryROS);

  // Publish TF
  quat_tf.setRPY(transform_to_be_mapped[0], transform_to_be_mapped[1], transform_to_be_mapped[2]);
  tf2::Transform t_odom_to_lidar = tf2::Transform(quat_tf, tf2::Vector3(transform_to_be_mapped[3], transform_to_be_mapped[4], transform_to_be_mapped[5]));
  tf2::TimePoint time_point = tf2_ros::fromRclcpp(time_laser_info_stamp_);
  tf2::Stamped<tf2::Transform> temp_odom_to_lidar(t_odom_to_lidar, time_point, odometry_frame_);
  geometry_msgs::msg::TransformStamped trans_odom_to_lidar;
  tf2::convert(temp_odom_to_lidar, trans_odom_to_lidar);
  trans_odom_to_lidar.header.frame_id = odometry_frame_;
  trans_odom_to_lidar.child_frame_id = base_link_frame_;
  br->sendTransform(trans_odom_to_lidar);

  // Publish odometry for ROS (incremental)
  static bool lastIncreOdomPubFlag = false;
  static nav_msgs::msg::Odometry laserOdomIncremental; // incremental odometry msg
  static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
  if (lastIncreOdomPubFlag == false)
  {
    lastIncreOdomPubFlag = true;
    laserOdomIncremental = laserOdometryROS;
    increOdomAffine = trans_to_affine3f(transform_to_be_mapped);
  } else {
    Eigen::Affine3f affineIncre = incremental_odometry_affine_front_.inverse() * incremental_odometry_affine_back_;
    increOdomAffine = increOdomAffine * affineIncre;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
    if (cloud_info_.imu_available == true)
    {
      if (std::abs(cloud_info_.imu_pitch_init) < 1.4)
      {
        double imuWeight = 0.1;
        tf2::Quaternion imuQuaternion;
        tf2::Quaternion transformQuaternion;
        double rollMid, pitchMid, yawMid;

        // slerp roll
        transformQuaternion.setRPY(roll, 0, 0);
        imuQuaternion.setRPY(cloud_info_.imu_roll_init, 0, 0);
        tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
        roll = rollMid;

        // slerp pitch
        transformQuaternion.setRPY(0, pitch, 0);
        imuQuaternion.setRPY(0, cloud_info_.imu_pitch_init, 0);
        tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
        pitch = pitchMid;
      }
    }
    laserOdomIncremental.header.stamp = time_laser_info_stamp_;
    laserOdomIncremental.header.frame_id = odometry_frame_;
    laserOdomIncremental.child_frame_id = base_link_frame_;
    laserOdomIncremental.pose.pose.position.x = x;
    laserOdomIncremental.pose.pose.position.y = y;
    laserOdomIncremental.pose.pose.position.z = z;
    tf2::Quaternion quat_tf;
    quat_tf.setRPY(roll, pitch, yaw);
    geometry_msgs::msg::Quaternion quat_msg;
    tf2::convert(quat_tf, quat_msg);
    laserOdomIncremental.pose.pose.orientation = quat_msg;
    if (is_degenerate_)
      laserOdomIncremental.pose.covariance[0] = 1;
    else
      laserOdomIncremental.pose.covariance[0] = 0;
  }
  pub_odom_laser_incremental->publish(laserOdomIncremental);
}


void FeatureMatching::publish_cloud(
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub,
  pcl::PointCloud<PointType>::Ptr thisCloud, rclcpp::Time thisStamp, std::string thisFrame)
{
  sensor_msgs::msg::PointCloud2 tempCloud;
  pcl::toROSMsg(*thisCloud, tempCloud);
  tempCloud.header.stamp = thisStamp;
  tempCloud.header.frame_id = thisFrame;
  if (thisPub->get_subscription_count() != 0)
    thisPub->publish(tempCloud);
}


void FeatureMatching::publish_frames(
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_key_poses,
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_recent_key_frames,
//  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub_cloud_registered,
  const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr & pub_path
  )
{
  if (cloud_key_poses_3d_->points.empty())
    return;
  // publish key poses
  publish_cloud(pub_key_poses, cloud_key_poses_3d_, time_laser_info_stamp_, odometry_frame_);
  // Publish surrounding key frames
  publish_cloud(pub_recent_key_frames, laser_cloud_surface_from_map_ds_,
                time_laser_info_stamp_, odometry_frame_);
  // publish registered key frame
  if (pub_recent_key_frames->get_subscription_count() != 0)
  {
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
    PointTypePose thisPose6D = trans_to_point_type_pose(transform_to_be_mapped);
    *cloudOut += *transform_point_cloud(laser_cloud_corner_last_ds_,  &thisPose6D);
    *cloudOut += *transform_point_cloud(laser_cloud_surface_last_ds_, &thisPose6D);
    publish_cloud(pub_recent_key_frames, cloudOut, time_laser_info_stamp_, odometry_frame_);
  }
  // publish registered high-res raw cloud
//  if (pub_cloud_registered->get_subscription_count() != 0)
//  {
//    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
////    pcl::fromROSMsg(cloud_info_.cloud_deskewed, *cloudOut);
//    pcl::fromROSMsg(cloud_deske0, *cloudOut);
//    PointTypePose thisPose6D = trans2PointTypePose(transform_to_be_mapped);
//    *cloudOut = *transform_point_cloud(cloudOut,  &thisPose6D);
//    publish_cloud(pub_cloud_registered, cloudOut, time_laser_info_stamp_, odometry_frame_);
//  }
  // publish path
  if (pub_path->get_subscription_count() != 0)
  {
    global_path_.header.stamp = time_laser_info_stamp_;
    global_path_.header.frame_id = odometry_frame_;
    pub_path->publish(global_path_);
  }
}





} // namespace loam_feature_localization