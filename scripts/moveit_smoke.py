"""Plan and physically execute a small joint move through MoveIt -> ROS action -> MuJoCo."""

import subprocess, time, json
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.action import ExecuteTrajectory

processes = []
rclpy.init()
node = Node("rusal_moveit_smoke")
states = []
node.create_subscription(JointState, "joint_states", lambda msg: states.append(msg), 10)
try:
    for name, cmd in [
        ("bridge", [".venv/bin/rusal-sim", "ros"]),
        ("moveit", [".venv/bin/python", "scripts/launch_moveit.py"]),
    ]:
        processes.append(
            subprocess.Popen(
                cmd,
                stdout=open("results/logs/moveit_smoke_" + name + ".txt", "w"),
                stderr=subprocess.STDOUT,
            )
        )
    client = node.create_client(GetMotionPlan, "plan_kinematic_path")
    assert client.wait_for_service(timeout_sec=20)
    start = time.monotonic()
    while time.monotonic() - start < 2 or not states:
        rclpy.spin_once(node, timeout_sec=0.1)
    request = GetMotionPlan.Request()
    q = request.motion_plan_request
    q.group_name = "manipulator"
    q.pipeline_id = "ompl"
    q.planner_id = "RRTConnect"
    q.allowed_planning_time = 5.0
    q.num_planning_attempts = 1
    q.start_state.joint_state = states[-1]
    q.start_state.is_diff = False
    constraint = Constraints()
    target = list(states[-1].position)
    target[0] += 0.02
    for name, value in zip(states[-1].name, target):
        jc = JointConstraint()
        jc.joint_name = name
        jc.position = value
        jc.tolerance_above = 0.001
        jc.tolerance_below = 0.001
        jc.weight = 1.0
        constraint.joint_constraints.append(jc)
    q.goal_constraints = [constraint]
    q.workspace_parameters.header.frame_id = "world"
    q.workspace_parameters.min_corner.x = q.workspace_parameters.min_corner.y = (
        q.workspace_parameters.min_corner.z
    ) = -4.0
    q.workspace_parameters.max_corner.x = q.workspace_parameters.max_corner.y = (
        q.workspace_parameters.max_corner.z
    ) = 4.0
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=12)
    assert future.done()
    response = future.result().motion_plan_response
    assert response.error_code.val == 1, f"Planning code {response.error_code.val}"
    action = ActionClient(node, ExecuteTrajectory, "execute_trajectory")
    assert action.wait_for_server(timeout_sec=5)
    goal = ExecuteTrajectory.Goal()
    goal.trajectory = response.trajectory
    sent = action.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, sent, timeout_sec=5)
    handle = sent.result()
    assert handle.accepted
    result = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result, timeout_sec=15)
    assert result.done()
    code = result.result().result.error_code.val
    assert code == 1, f"Execution code {code}"
    report = dict(
        passed=True,
        planning_code=response.error_code.val,
        execution_code=code,
        trajectory_points=len(response.trajectory.joint_trajectory.points),
        scope="small joint move; no grasp planning",
    )
    open("results/reports/moveit_smoke.json", "w").write(json.dumps(report, indent=2))
    print(report)
finally:
    for proc in processes:
        proc.terminate()
    for proc in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    node.destroy_node()
    rclpy.shutdown()
