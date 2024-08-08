import os

from ament_index_python import get_package_share_directory

import launch

from launch.actions import DeclareLaunchArgument

from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description."""
    path_package = get_package_share_directory('loam_feature_localization')

    loam_feature_localization_file_param = os.path.join(path_package, 'config/loam_feature_localization.param.yaml')
    loam_feature_localization_node = Node(
        package='loam_feature_localization',
        executable='loam_feature_localization_node',
        parameters=[loam_feature_localization_file_param]
    )


    return launch.LaunchDescription(
        [loam_feature_localization_node])
