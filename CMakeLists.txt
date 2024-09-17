cmake_minimum_required(VERSION 3.14)
project(loam_feature_localization)

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
        ${PCL_INCLUDE_DIRS}
        ${pcl_conversions_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIRS}
        ${OpenMP_INCLUDE_DIRS}
        ${cv_bridge_INCLUDE_DIRS})


set(LOAM_FEATURE_LOCALIZATION_SRC
        src/loam_feature_localization.cpp
        src/feature_extraction.cpp
        src/image_projection.cpp
        src/imu_preintegration.cpp
        src/feature_matching.cpp
        src/utils.cpp)
set(LOAM_FEATURE_LOCALIZATION_HEADERS
        include/loam_feature_localization/loam_feature_localization.hpp
        include/loam_feature_localization/feature_extraction.hpp
        include/loam_feature_localization/image_projection.hpp
        include/loam_feature_localization/imu_preintegration.hpp
        include/loam_feature_localization/feature_matching.hpp
        include/loam_feature_localization/stop_watch.hpp
        include/loam_feature_localization/utils.hpp)


ament_auto_add_library(${PROJECT_NAME} SHARED
        ${LOAM_FEATURE_LOCALIZATION_HEADERS}
        ${LOAM_FEATURE_LOCALIZATION_SRC})
target_link_libraries(${PROJECT_NAME}
        ${PCL_LIBRARIES}
        ${pcl_conversions_LIBRARIES}
        ${OpenCV_LIBRARIES}
        gtsam
        ${OpenMP_LIBRARIES}
        ${cv_bridge_LIBRARIES})

rclcpp_components_register_node(${PROJECT_NAME}
        PLUGIN "loam_feature_localization::LoamFeatureLocalization"
        EXECUTABLE ${PROJECT_NAME}_node
)

#if(BUILD_TESTING)
#
#endif()

ament_auto_package(
        INSTALL_TO_SHARE
        launch
        config
)
