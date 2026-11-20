"""End-to-end ROS publisher/service smoke check with live MuJoCo bridge subprocess."""

import subprocess, time, json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from std_msgs.msg import String
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

proc = subprocess.Popen(
    [".venv/bin/rusal-sim", "ros"],
    stdout=open("results/logs/ros_bridge.txt", "w"),
    stderr=subprocess.STDOUT,
)
rclpy.init()
node = Node("rusal_smoke")
counts = {"joint_states": 0, "gripper": 0}
last = {}


def js(msg):
    counts["joint_states"] += 1
    last["joints"] = msg.name
    last["positions"] = list(msg.position)
    last["finite"] = all(abs(p) < 100 for p in msg.position)


def state(msg):
    counts["gripper"] += 1
    last["gripper"] = json.loads(msg.data)


node.create_subscription(JointState, "joint_states", js, 10)
node.create_subscription(String, "rusal/gripper_state", state, 10)
client = node.create_client(SetBool, "rusal/vacuum")
try:
    assert client.wait_for_service(timeout_sec=15), "Bridge service missing"
    req = SetBool.Request()
    req.data = True
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5)
    assert future.done() and future.result().success
    start = time.monotonic()
    while time.monotonic() - start < 2:
        rclpy.spin_once(node, timeout_sec=0.1)
    assert (
        counts["joint_states"] > 5
        and counts["gripper"] > 5
        and last["finite"]
        and last["gripper"]["vacuum_on"]
    )
    action = ActionClient(node, FollowJointTrajectory, "rusal/follow_joint_trajectory")
    assert action.wait_for_server(timeout_sec=5)
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = last["joints"]
    point = JointTrajectoryPoint()
    point.positions = last["positions"].copy()
    point.positions[0] += 0.01
    point.time_from_start.sec = 2
    goal.trajectory.points = [point]
    sent = action.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, sent, timeout_sec=5)
    handle = sent.result()
    assert handle.accepted
    completed = handle.get_result_async()
    rclpy.spin_until_future_complete(node, completed, timeout_sec=8)
    assert completed.done() and completed.result().result.error_code == 0, str(completed.result())
    report = {
        "trajectory_action": True,
        "passed": True,
        "messages": counts,
        "joints": last["joints"],
        "vacuum_service": True,
    }
    open("results/reports/ros_smoke.json", "w").write(json.dumps(report, indent=2))
    print(report)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    node.destroy_node()
    rclpy.shutdown()
