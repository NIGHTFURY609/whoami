from glob import glob

from setuptools import find_packages, setup

package_name = "ugv_localization"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/scripts", glob("scripts/*.sh")),
        # Architecture §7: camera calibration lives in ugv_nav/config/cameras/ (Dev 2 owned).
        (f"share/{package_name}/config/cameras", glob("../config/cameras/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jeswin-christie",
    maintainer_email="dev@neogenmedia.com",
    description="Dev 2 localization: RTAB-Map mono + wheel odom, TF chain, pose validity.",
    license="Proprietary",
    extras_require={"test": ["pytest"]},  # colcon picks pytest from here (tests_require is gone)
    entry_points={
        "console_scripts": [
            "odom_tf_bridge = ugv_localization.nodes.odom_tf_bridge:main",
            "pose_validity_node = ugv_localization.nodes.pose_validity_node:main",
            "tf_rate_check = ugv_localization.nodes.tf_rate_check:main",
            "drift_eval = ugv_localization.nodes.drift_eval:main",
            "mode_cli = ugv_localization.nodes.mode_cli:main",
            "camera_info_to_yaml = ugv_localization.nodes.camera_info_to_yaml:main",
            "drift_report = ugv_localization.tools.drift_report:main",
            "check_rtabmap_params = ugv_localization.tools.check_rtabmap_params:main",
        ],
    },
)
