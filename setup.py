import os
from glob import glob

from setuptools import find_packages, setup

package_name = "urc_rock_detection"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Urwa Eman",
    maintainer_email="urwaeman64@gmail.com",
    description="OpenCV rock and crater detection node publishing hazards to the Nav2 local costmap",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rock_detection_node = urc_rock_detection.rock_detection_node:main",
            "evaluate = urc_rock_detection.evaluate:main",
            "fake_camera_node = urc_rock_detection.fake_camera_node:main",
            "make_dummy_dataset = urc_rock_detection.make_dummy_dataset:main",
        ],
    },
)
