from launch import LaunchService
import runpy
from rusal_sim import ROOT

module = runpy.run_path(str(ROOT / "src/rusal_motion_planning/launch/moveit.launch.py"))
service = LaunchService()
service.include_launch_description(module["generate_launch_description"]())
service.run()
