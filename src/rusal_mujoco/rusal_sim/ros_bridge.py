"""ROS 2 Jazzy bridge to one MuJoCo world; finite position drives, simulated clock.
This Python bridge is separate from the future mujoco_ros2_control hardware plugin.
"""

import json
import threading
import time
import xml.etree.ElementTree as E
import numpy as np
import mujoco
from . import ROOT, configuration
from .vacuum import Vacuum


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionServer, GoalResponse, CancelResponse
    from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
    from rclpy.callback_groups import ReentrantCallbackGroup
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseArray, Pose, TransformStamped
    from std_msgs.msg import String, Float64MultiArray
    from std_srvs.srv import SetBool
    from control_msgs.action import FollowJointTrajectory
    from rosgraph_msgs.msg import Clock
    from tf2_ros import TransformBroadcaster
    from builtin_interfaces.msg import Time
    from .robot import URDF

    class Bridge(Node):
        def __init__(self):
            super().__init__("rusal_mujoco")
            self.m = mujoco.MjModel.from_xml_path(str(ROOT / "src/rusal_description/mjcf/cell.xml"))
            self.d = mujoco.MjData(self.m)
            self.cfg = configuration()
            self.vac = Vacuum(self.m, self.cfg)
            self.lock = threading.RLock()
            self.vac_on = False
            self.close = False
            self.deploy = False
            self.names = [f"joint_{i}" for i in range(1, 7)]
            self.ids = [self.m.joint(n).id for n in self.names]
            self.qa = self.m.jnt_qposadr[self.ids]
            self.va = self.m.jnt_dofadr[self.ids]
            self.targets = np.array([0, -1.2, 1.2, 0, 1, 0.0])
            self.d.qpos[self.qa] = self.targets
            self.trajectory = None
            self.active_goal = False
            u = E.parse(URDF).getroot()
            self.vel_limits = np.array(
                [float(u.find(f"joint[@name='{n}']/limit").get("velocity")) for n in self.names]
            )
            self.js = self.create_publisher(JointState, "joint_states", 10)
            self.poses = self.create_publisher(PoseArray, "rusal/ingots", 10)
            self.meta = self.create_publisher(String, "rusal/ingot_metadata", 10)
            self.diag = self.create_publisher(String, "rusal/gripper_state", 10)
            self.clock = self.create_publisher(Clock, "clock", 10)
            self.tf = TransformBroadcaster(self)
            self.create_service(SetBool, "rusal/vacuum", self.vacuum_service)
            self.create_service(SetBool, "rusal/jaws_close", self.jaw_service)
            self.create_service(SetBool, "rusal/jaws_deploy", self.deploy_service)
            self.create_subscription(Float64MultiArray, "rusal/joint_targets", self.command, 10)
            self.server = ActionServer(
                self,
                FollowJointTrajectory,
                "rusal/follow_joint_trajectory",
                execute_callback=self.execute,
                goal_callback=self.goal,
                cancel_callback=lambda _: CancelResponse.ACCEPT,
                callback_group=ReentrantCallbackGroup(),
            )
            self.create_timer(0.01, self.tick)

        def vacuum_service(self, req, res):
            self.vac_on = req.data
            res.success = True
            return res

        def jaw_service(self, req, res):
            self.close = req.data
            res.success = True
            return res

        def deploy_service(self, req, res):
            self.deploy = req.data
            res.success = True
            return res

        def command(self, msg):
            if len(msg.data) == 6 and np.all(np.isfinite(msg.data)):
                with self.lock:
                    self.targets = np.clip(
                        msg.data, self.m.jnt_range[self.ids, 0], self.m.jnt_range[self.ids, 1]
                    )

        def goal(self, req):
            tr = req.trajectory
            if tr.joint_names != self.names or not tr.points or self.active_goal:
                return GoalResponse.REJECT
            prev = np.zeros(6)
            last = -1
            for index, p in enumerate(tr.points):
                t = p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
                q = np.array(p.positions)
                if (
                    len(q) != 6
                    or not np.all(np.isfinite(q))
                    or t < 0
                    or (index > 0 and t <= last)
                    or np.any(q < self.m.jnt_range[self.ids, 0])
                    or np.any(q > self.m.jnt_range[self.ids, 1])
                ):
                    return GoalResponse.REJECT
                if index > 0 and np.any(np.abs(q - prev) / (t - last) > self.vel_limits):
                    return GoalResponse.REJECT
                if index == 0 and t == 0 and np.max(np.abs(q - self.d.qpos[self.qa])) > 0.03:
                    return GoalResponse.REJECT
                prev = q
                last = t
            return GoalResponse.ACCEPT

        def execute(self, handle):
            tr = handle.request.trajectory
            result = FollowJointTrajectory.Result()
            with self.lock:
                self.active_goal = True
                start = self.d.time
                points = [
                    p
                    for p in tr.points
                    if p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 > 0
                ]
                ts = np.array(
                    [0] + [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in points]
                )
                qs = np.array([self.d.qpos[self.qa].copy()] + [p.positions for p in points])
                if np.any(np.abs(np.diff(qs, axis=0)) / np.diff(ts)[:, None] > self.vel_limits):
                    self.active_goal = False
                    result.error_code = result.INVALID_GOAL
                    handle.abort()
                    return result
                self.trajectory = (start, ts, qs)
            while rclpy.ok():
                with self.lock:
                    if handle.is_cancel_requested:
                        self.trajectory = None
                        self.targets = self.d.qpos[self.qa].copy()
                        self.active_goal = False
                        handle.canceled()
                        return result
                    error = np.max(np.abs(self.d.qpos[self.qa] - qs[-1]))
                    done = self.d.time - start >= ts[-1] and (
                        error < 0.03 or self.d.time - start >= ts[-1] + 1
                    )
                if done:
                    break
                time.sleep(0.02)
            with self.lock:
                self.trajectory = None
                self.active_goal = False
            if error > 0.03:
                result.error_code = result.GOAL_TOLERANCE_VIOLATED
                handle.abort()
            else:
                result.error_code = result.SUCCESSFUL
                handle.succeed()
            return result

        def tick(self):
            with self.lock:
                m, d = self.m, self.d
                for _ in range(round(0.01 / m.opt.timestep)):
                    if self.trajectory:
                        start, ts, qs = self.trajectory
                        self.targets = np.array(
                            [np.interp(d.time - start, ts, qs[:, i]) for i in range(6)]
                        )
                    for name, q in zip(self.names, self.targets):
                        d.ctrl[m.actuator(name + "_position").id] = (
                            q
                            + d.qfrc_bias[m.jnt_dofadr[m.joint(name).id]]
                            / self.cfg["robot"]["position_kp"]
                        )
                    self.vac.update(m, d, self.vac_on)
                    for side in ["left", "right"]:
                        d.ctrl[m.actuator(f"deploy_motor_{side}").id] = (
                            self.cfg["jaws"]["deploy_travel"] if self.deploy else 0
                        )
                        j = m.joint(f"jaw_{side}")
                        target = self.cfg["jaws"]["closed_travel"] if self.close else 0
                        d.ctrl[m.actuator(f"jaw_motor_{side}").id] = np.clip(
                            self.cfg["jaws"]["kp"] * (target - d.qpos[j.qposadr[0]])
                            - self.cfg["jaws"]["damping"] * d.qvel[j.dofadr[0]],
                            -self.cfg["jaws"]["force_per_jaw"],
                            self.cfg["jaws"]["force_per_jaw"],
                        )
                    mujoco.mj_step(m, d)
                stamp = Time(sec=int(d.time), nanosec=int(d.time % 1 * 1e9))
                clock = Clock()
                clock.clock = stamp
                self.clock.publish(clock)
                msg = JointState()
                msg.header.stamp = stamp
                msg.name = self.names
                msg.position = d.qpos[self.qa].tolist()
                msg.velocity = d.qvel[self.va].tolist()
                msg.effort = d.qfrc_actuator[self.va].tolist()
                self.js.publish(msg)
                poses = PoseArray()
                poses.header.stamp = stamp
                poses.header.frame_id = "world"
                metadata = []
                for body in range(m.nbody):
                    name = m.body(body).name
                    if not name.startswith("ingot"):
                        continue
                    po = Pose()
                    po.position.x, po.position.y, po.position.z = map(float, d.xpos[body])
                    q = d.xquat[body]
                    po.orientation.w, po.orientation.x, po.orientation.y, po.orientation.z = map(
                        float, q
                    )
                    poses.poses.append(po)
                    metadata.append(
                        dict(
                            id=name,
                            dimensions=[
                                self.cfg["ingot"][k] for k in ["length", "width", "thickness"]
                            ],
                            mass=float(m.body_mass[body]),
                            ribbed_normal=d.xmat[body].reshape(3, 3)[:, 2].tolist(),
                        )
                    )
                    tf = TransformStamped()
                    tf.header = poses.header
                    tf.child_frame_id = name
                    (
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z,
                    ) = map(float, d.xpos[body])
                    tf.transform.rotation = po.orientation
                    self.tf.sendTransform(tf)
                self.poses.publish(poses)
                meta = String()
                meta.data = json.dumps(metadata)
                self.meta.publish(meta)
                jaw_positions = [
                    float(d.qpos[m.joint("jaw_" + side).qposadr[0]]) for side in ["left", "right"]
                ]
                jaw_forces = [0.0, 0.0]
                cf = np.zeros(6)
                for ci, contact in enumerate(d.contact):
                    mujoco.mj_contactForce(m, d, ci, cf)
                    names = [m.geom(contact.geom1).name, m.geom(contact.geom2).name]
                    for ji, side in enumerate(["left", "right"]):
                        if "jaw_pad_" + side in names:
                            jaw_forces[ji] += abs(float(cf[0]))
                st = String()
                st.data = json.dumps(
                    dict(
                        jaw_positions=jaw_positions,
                        jaw_forces_n=jaw_forces,
                        pressure_pa=self.vac.pressures.tolist(),
                        seal_quality=self.vac.quality.tolist(),
                        capacity_n=self.vac.capacity.tolist(),
                        source="model estimates",
                        vacuum_on=self.vac_on,
                    )
                )
                self.diag.publish(st)

    rclpy.init()
    node = Bridge()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
