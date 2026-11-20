"""MoveIt talks to the same MuJoCo world through the tested Python trajectory action.
No fake hardware, no second simulator. Tool collision geometry and task pipeline
must be completed before executing close-clearance heap plans.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
import xml.etree.ElementTree as E
from rusal_sim import ROOT


def generate_launch_description():
    description = (ROOT / "src/rusal_description/urdf/kr50_r2500.urdf").read_text()
    u = E.fromstring(description)
    # Resolve ROS package mesh URIs for the uninstalled upstream checkout.
    description = description.replace(
        "package://", "file://" + str(ROOT / "src/third_party/kuka_robot_descriptions") + "/"
    )
    semantic = """<robot name="kr50_r2500"><virtual_joint name="world_joint" type="fixed" parent_frame="world" child_link="base_link"/><group name="manipulator"><chain base_link="base_link" tip_link="tool0"/></group><disable_collisions link1="base_link" link2="link_1" reason="Adjacent"/><disable_collisions link1="link_1" link2="link_2" reason="Adjacent"/><disable_collisions link1="link_2" link2="link_3" reason="Adjacent"/><disable_collisions link1="link_3" link2="link_4" reason="Adjacent"/><disable_collisions link1="link_4" link2="link_5" reason="Adjacent"/><disable_collisions link1="link_5" link2="link_6" reason="Adjacent"/></robot>"""
    limits = {
        j.get("name"): {
            "has_velocity_limits": True,
            "max_velocity": float(j.find("limit").get("velocity")),
            "has_acceleration_limits": True,
            "max_acceleration": 1.0,
        }
        for j in u.findall("joint")
        if j.get("type") == "revolute"
    }
    names = [f"joint_{i}" for i in range(1, 7)]
    params = {
        "robot_description": description,
        "robot_description_semantic": semantic,
        "use_sim_time": True,
        "robot_description_kinematics": {
            "manipulator": {
                "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
                "kinematics_solver_timeout": 0.1,
            }
        },
        "robot_description_planning": {"joint_limits": limits},
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": {
            "planning_plugins": ["ompl_interface/OMPLPlanner"],
            "request_adapters": [
                "default_planning_request_adapters/ResolveConstraintFrames",
                "default_planning_request_adapters/ValidateWorkspaceBounds",
                "default_planning_request_adapters/CheckStartStateBounds",
                "default_planning_request_adapters/CheckStartStateCollision",
            ],
            "response_adapters": [
                "default_planning_response_adapters/AddTimeOptimalParameterization",
                "default_planning_response_adapters/ValidateSolution",
            ],
            "manipulator": {"planner_configs": ["RRTConnect"]},
            "planner_configs": {"RRTConnect": {"type": "geometric::RRTConnect", "range": 0.1}},
        },
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "moveit_simple_controller_manager": {
            "controller_names": ["rusal"],
            "rusal": {
                "type": "FollowJointTrajectory",
                "action_ns": "follow_joint_trajectory",
                "default": True,
                "joints": names,
            },
        },
        "trajectory_execution.allowed_start_tolerance": 0.03,
        "publish_robot_description_semantic": True,
    }
    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": description, "use_sim_time": True}],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                arguments=[
                    "--x",
                    "-1.2",
                    "--y",
                    "0",
                    "--z",
                    "0",
                    "--frame-id",
                    "world",
                    "--child-frame-id",
                    "base_link",
                ],
            ),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                parameters=[params],
                output="screen",
            ),
        ]
    )
