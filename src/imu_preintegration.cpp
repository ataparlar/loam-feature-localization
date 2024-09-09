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

#include "loam_feature_localization/imu_preintegration.hpp"

#include "tf2_eigen/tf2_eigen.hpp"

#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using gtsam::symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

namespace loam_feature_localization
{

TransformFusion::TransformFusion( const Utils::SharedPtr & utils,
  std::string base_link_frame, std::string lidar_frame, std::string odometry_frame)
{
  utils_ = utils;
  base_link_frame_ = base_link_frame;
  odometry_frame_ = odometry_frame;
  lidar_frame_ = lidar_frame;
}

Eigen::Isometry3d TransformFusion::odom2affine(nav_msgs::msg::Odometry odom)
{
  tf2::Transform t;
  tf2::fromMsg(odom.pose.pose, t);
  return tf2::transformToEigen(tf2::toMsg(t));
}

void TransformFusion::lidar_odometry_handler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
{
//  std::lock_guard<std::mutex> lock(mtx);

  lidarOdomAffine = odom2affine(*odomMsg);

  lidarOdomTime = utils_->stamp2Sec(odomMsg->header.stamp);
}

void TransformFusion::imu_odometry_handler(
  const nav_msgs::msg::Odometry::SharedPtr odomMsg, rclcpp::Logger logger_,
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry,
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubImuPath)
{
//  std::lock_guard<std::mutex> lock(mtx);

  imuOdomQueue.push_back(*odomMsg);

  // get latest odometry (at current IMU stamp)
  if (lidarOdomTime == -1) return;
  while (!imuOdomQueue.empty()) {
    if (utils_->stamp2Sec(imuOdomQueue.front().header.stamp) <= lidarOdomTime)
      imuOdomQueue.pop_front();
    else
      break;
  }
  Eigen::Isometry3d imuOdomAffineFront = odom2affine(imuOdomQueue.front());
  Eigen::Isometry3d imuOdomAffineBack = odom2affine(imuOdomQueue.back());
  Eigen::Isometry3d imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
  Eigen::Isometry3d imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
  auto t = tf2::eigenToTransform(imuOdomAffineLast);
  tf2::Stamped<tf2::Transform> tCur;
  tf2::convert(t, tCur);

  // publish latest odometry
  nav_msgs::msg::Odometry laserOdometry = imuOdomQueue.back();
  laserOdometry.pose.pose.position.x = t.transform.translation.x;
  laserOdometry.pose.pose.position.y = t.transform.translation.y;
  laserOdometry.pose.pose.position.z = t.transform.translation.z;
  laserOdometry.pose.pose.orientation = t.transform.rotation;
  pubImuOdometry->publish(laserOdometry);

  // publish tf
  if (lidar_frame_ != base_link_frame_) {
    try {
      tf2::fromMsg(
        tfBuffer->lookupTransform(lidar_frame_, base_link_frame_, rclcpp::Time(0)), lidar2Baselink);
    } catch (tf2::TransformException & ex) {
      RCLCPP_ERROR(logger_, "%s", ex.what());
    }
    tf2::Stamped<tf2::Transform> tb(
      tCur * lidar2Baselink, tf2_ros::fromMsg(odomMsg->header.stamp), odometry_frame_);
    tCur = tb;
  }
  geometry_msgs::msg::TransformStamped ts;
  tf2::convert(tCur, ts);
  ts.header.frame_id = odometry_frame_;
  ts.child_frame_id = base_link_frame_;
  tfBroadcaster->sendTransform(ts);

  // publish IMU path
  static nav_msgs::msg::Path imuPath;
  static double last_path_time = -1;
  double imuTime = utils_->stamp2Sec(imuOdomQueue.back().header.stamp);
  if (imuTime - last_path_time > 0.1) {
    last_path_time = imuTime;
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
    pose_stamped.header.frame_id = odometry_frame_;
    pose_stamped.pose = laserOdometry.pose.pose;
    imuPath.poses.push_back(pose_stamped);
    while (!imuPath.poses.empty() &&
           utils_->stamp2Sec(imuPath.poses.front().header.stamp) < lidarOdomTime - 1.0)
      imuPath.poses.erase(imuPath.poses.begin());
    if (pubImuPath->get_subscription_count() != 0) {
      imuPath.header.stamp = imuOdomQueue.back().header.stamp;
      imuPath.header.frame_id = odometry_frame_;
      pubImuPath->publish(imuPath);
    }
  }
}

ImuPreintegration::ImuPreintegration(  const Utils::SharedPtr & utils, std::string odometry_frame,
  float lidar_imu_x, float lidar_imu_y, float lidar_imu_z, float imu_gravity, float imu_acc_noise,
  float imu_acc_bias, float imu_gyro_noise, float imu_gyro_bias)
{
  utils_ = utils;

//  base_link_frame_ = base_link_frame;
  odometry_frame_ = odometry_frame;
//  lidar_frame_ = lidar_frame;
  lidar_imu_x_ = lidar_imu_x;
  lidar_imu_y_ = lidar_imu_y;
  lidar_imu_z_ = lidar_imu_z;

  imu2Lidar = gtsam::Pose3(
    gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-lidar_imu_x_, -lidar_imu_y_, -lidar_imu_z_));
  lidar2Imu =
    gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(lidar_imu_x_, lidar_imu_y_, lidar_imu_z_));

  boost::shared_ptr<gtsam::PreintegrationParams> p =
    gtsam::PreintegrationParams::MakeSharedU(imu_gravity);
  p->accelerometerCovariance =
    gtsam::Matrix33::Identity(3, 3) * pow(imu_acc_noise, 2);  // acc white noise in continuous
  p->gyroscopeCovariance =
    gtsam::Matrix33::Identity(3, 3) * pow(imu_gyro_noise, 2);  // gyro white noise in continuous
  p->integrationCovariance =
    gtsam::Matrix33::Identity(3, 3) *
    pow(1e-4, 2);  // error committed in integrating position from velocities
  gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());
  ;  // assume zero initial bias

  priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished());  // rad,rad,rad,m, m, m
  priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);             // m/s
  priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);  // 1e-2 ~ 1e-3 seems to be good
  correctionNoise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());  // rad,rad,rad,m, m, m
  correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished());  // rad,rad,rad,m, m, m
  noiseModelBetweenBias = (gtsam::Vector(6) << imu_acc_bias, imu_acc_bias, imu_acc_bias,
                           imu_gyro_bias, imu_gyro_bias, imu_gyro_bias)
                            .finished();

  imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(
    p, prior_imu_bias);  // setting up the IMU integration for IMU message thread
  imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(
    p, prior_imu_bias);  // setting up the IMU integration for optimization
}

void ImuPreintegration::reset_optimization()
{
  gtsam::ISAM2Params optParameters;
  optParameters.relinearizeThreshold = 0.1;
  optParameters.relinearizeSkip = 1;
  optimizer = gtsam::ISAM2(optParameters);

  gtsam::NonlinearFactorGraph newGraphFactors;
  graphFactors = newGraphFactors;

  gtsam::Values NewGraphValues;
  graphValues = NewGraphValues;
}

void ImuPreintegration::reset_params()
{
  lastImuT_imu = -1;
  doneFirstOpt = false;
  systemInitialized = false;
}

void ImuPreintegration::odometry_handler(const nav_msgs::msg::Odometry::SharedPtr odomMsg, rclcpp::Logger logger_)
{
//  std::lock_guard<std::mutex> lock(mtx);

  double currentCorrectionTime = utils_->stamp2Sec(odomMsg->header.stamp);

  // make sure we have imu data to integrate
  if (imuQueOpt.empty()) return;

  float p_x = odomMsg->pose.pose.position.x;
  float p_y = odomMsg->pose.pose.position.y;
  float p_z = odomMsg->pose.pose.position.z;
  float r_x = odomMsg->pose.pose.orientation.x;
  float r_y = odomMsg->pose.pose.orientation.y;
  float r_z = odomMsg->pose.pose.orientation.z;
  float r_w = odomMsg->pose.pose.orientation.w;
  bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
  gtsam::Pose3 lidarPose =
    gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

  // 0. initialize system
  if (systemInitialized == false) {
    reset_optimization();

    // pop old IMU message
    while (!imuQueOpt.empty()) {
      if (utils_->stamp2Sec(imuQueOpt.front().header.stamp) < currentCorrectionTime - delta_t) {
        lastImuT_opt = utils_->stamp2Sec(imuQueOpt.front().header.stamp);
        imuQueOpt.pop_front();
      } else
        break;
    }
    // initial pose
    prevPose_ = lidarPose.compose(lidar2Imu);
    gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
    graphFactors.add(priorPose);
    // initial velocity
    prevVel_ = gtsam::Vector3(0, 0, 0);
    gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
    graphFactors.add(priorVel);
    // initial bias
    prevBias_ = gtsam::imuBias::ConstantBias();
    gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
    graphFactors.add(priorBias);
    // add values
    graphValues.insert(X(0), prevPose_);
    graphValues.insert(V(0), prevVel_);
    graphValues.insert(B(0), prevBias_);
    // optimize once
    optimizer.update(graphFactors, graphValues);
    graphFactors.resize(0);
    graphValues.clear();

    imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
    imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

    key = 1;
    systemInitialized = true;
    return;
  }

  // reset graph for speed
  if (key == 100) {
    // get updated noise before reset
    gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise =
      gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
    gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise =
      gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
    gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise =
      gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));
    // reset graph
    reset_optimization();
    // add pose
    gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
    graphFactors.add(priorPose);
    // add velocity
    gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
    graphFactors.add(priorVel);
    // add bias
    gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
    graphFactors.add(priorBias);
    // add values
    graphValues.insert(X(0), prevPose_);
    graphValues.insert(V(0), prevVel_);
    graphValues.insert(B(0), prevBias_);
    // optimize once
    optimizer.update(graphFactors, graphValues);
    graphFactors.resize(0);
    graphValues.clear();

    key = 1;
  }

  // 1. integrate imu data and optimize
  while (!imuQueOpt.empty()) {
    // pop and integrate imu data that is between two optimizations
    sensor_msgs::msg::Imu * thisImu = &imuQueOpt.front();
    double imuTime = utils_->stamp2Sec(thisImu->header.stamp);
    if (imuTime < currentCorrectionTime - delta_t) {
      double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
      imuIntegratorOpt_->integrateMeasurement(
        gtsam::Vector3(
          thisImu->linear_acceleration.x, thisImu->linear_acceleration.y,
          thisImu->linear_acceleration.z),
        gtsam::Vector3(
          thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z),
        dt);

      lastImuT_opt = imuTime;
      imuQueOpt.pop_front();
    } else
      break;
  }
  // add imu factor to graph
  const gtsam::PreintegratedImuMeasurements & preint_imu =
    dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*imuIntegratorOpt_);
  gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
  graphFactors.add(imu_factor);
  // add imu bias between factor
  graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
    B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
    gtsam::noiseModel::Diagonal::Sigmas(
      sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
  // add pose factor
  gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
  gtsam::PriorFactor<gtsam::Pose3> pose_factor(
    X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
  graphFactors.add(pose_factor);
  // insert predicted values
  gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
  graphValues.insert(X(key), propState_.pose());
  graphValues.insert(V(key), propState_.v());
  graphValues.insert(B(key), prevBias_);
  // optimize
  optimizer.update(graphFactors, graphValues);
  optimizer.update();
  graphFactors.resize(0);
  graphValues.clear();
  // Overwrite the beginning of the preintegration for the next step.
  gtsam::Values result = optimizer.calculateEstimate();
  prevPose_ = result.at<gtsam::Pose3>(X(key));
  prevVel_ = result.at<gtsam::Vector3>(V(key));
  prevState_ = gtsam::NavState(prevPose_, prevVel_);
  prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));
  // Reset the optimization preintegration object.
  imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
  // check optimization
  if (failure_detection(prevVel_, prevBias_, logger_)) {
    reset_params();
    return;
  }

  // 2. after optiization, re-propagate imu odometry preintegration
  prevStateOdom = prevState_;
  prevBiasOdom = prevBias_;
  // first pop imu message older than current correction data
  double lastImuQT = -1;
  while (!imuQueImu.empty() &&
         utils_->stamp2Sec(imuQueImu.front().header.stamp) < currentCorrectionTime - delta_t) {
    lastImuQT = utils_->stamp2Sec(imuQueImu.front().header.stamp);
    imuQueImu.pop_front();
  }
  // repropogate
  if (!imuQueImu.empty()) {
    // reset bias use the newly optimized bias
    imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
    // integrate imu message from the beginning of this optimization
    for (int i = 0; i < (int)imuQueImu.size(); ++i) {
      sensor_msgs::msg::Imu * thisImu = &imuQueImu[i];
      double imuTime = utils_->stamp2Sec(thisImu->header.stamp);
      double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);

      imuIntegratorImu_->integrateMeasurement(
        gtsam::Vector3(
          thisImu->linear_acceleration.x, thisImu->linear_acceleration.y,
          thisImu->linear_acceleration.z),
        gtsam::Vector3(
          thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z),
        dt);
      lastImuQT = imuTime;
    }
  }

  ++key;
  doneFirstOpt = true;
}

bool ImuPreintegration::failure_detection(
  const gtsam::Vector3 & velCur, const gtsam::imuBias::ConstantBias & biasCur,
  const rclcpp::Logger & logger_)
{
  Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
  if (vel.norm() > 30) {
    RCLCPP_WARN(logger_, "Large velocity, reset IMU-preintegration!");
    return true;
  }

  Eigen::Vector3f ba(
    biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
  Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
  if (ba.norm() > 1.0 || bg.norm() > 1.0) {
    RCLCPP_WARN(logger_, "Large bias, reset IMU-preintegration!");
    return true;
  }

  return false;
}

void ImuPreintegration::imu_handler(
  const sensor_msgs::msg::Imu::SharedPtr imu_raw,
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry)
{
//  std::lock_guard<std::mutex> lock(mtx);

  sensor_msgs::msg::Imu thisImu = utils_->imuConverter(*imu_raw);

  imuQueOpt.push_back(thisImu);
  imuQueImu.push_back(thisImu);

  if (doneFirstOpt == false) return;

  double imuTime = utils_->stamp2Sec(thisImu.header.stamp);
  double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
  lastImuT_imu = imuTime;

  // integrate this single imu message
  imuIntegratorImu_->integrateMeasurement(
    gtsam::Vector3(
      thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
    gtsam::Vector3(
      thisImu.angular_velocity.x, thisImu.angular_velocity.y, thisImu.angular_velocity.z),
    dt);

  // predict odometry
  gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

  // publish odometry
  auto odometry = nav_msgs::msg::Odometry();
  odometry.header.stamp = thisImu.header.stamp;
  odometry.header.frame_id = odometry_frame_;
  odometry.child_frame_id = "odom_imu";

  // transform imu pose to ldiar
  gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
  gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

  odometry.pose.pose.position.x = lidarPose.translation().x();
  odometry.pose.pose.position.y = lidarPose.translation().y();
  odometry.pose.pose.position.z = lidarPose.translation().z();
  odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
  odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
  odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
  odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();

  odometry.twist.twist.linear.x = currentState.velocity().x();
  odometry.twist.twist.linear.y = currentState.velocity().y();
  odometry.twist.twist.linear.z = currentState.velocity().z();
  odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
  odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
  odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
  pubImuOdometry->publish(odometry);
}

}  // namespace loam_feature_localization
