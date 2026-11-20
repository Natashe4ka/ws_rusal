from setuptools import setup

setup(
    name="rusal_mujoco",
    version="0.1.0",
    description="MuJoCo contact dynamics for robotic ingot handling and palletizing",
    maintainer="rusal-sim maintainers",
    maintainer_email="safonova.nk@gmail.com",
    license="Apache-2.0",
    packages=["rusal_sim"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/rusal_mujoco"]),
        ("share/rusal_mujoco", ["package.xml"]),
    ],
    entry_points={"console_scripts": ["bridge=rusal_sim.ros_bridge:main"]},
)
