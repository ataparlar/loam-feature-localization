import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os
from ament_index_python import get_package_share_directory


def generate_launch_description():
    path_package = get_package_share_directory('loam_feature_localization')

    loam_feature_localization_file_param = os.path.join(path_package, 'config/loam_feature_localization.param.yaml')
    container = ComposableNodeContainer(
        name='loam_feature_loc',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='loam_feature_localization',
                plugin='loam_feature_localization::LoamFeatureLocalization',
                name='loam_feature_localization',
                parameters=[loam_feature_localization_file_param])
        ],
        output='both',
    )

    return launch.LaunchDescription([container])