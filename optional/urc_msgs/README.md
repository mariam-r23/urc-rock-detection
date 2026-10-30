# urc_msgs (minimal, optional)

The Crucible resource package only ships `urc_msgs/msg/DetectionBoundingBox.msg` (no `CMakeLists.txt` /
`package.xml`), so it cannot be built on its own. This folder wraps that **unchanged** .msg so
`urc_rock_detection` can publish `urc_msgs/DetectionBoundingBox`.

Use it ONLY if the real RoboJackets `urc_msgs` is not already in your workspace (two packages with the same name conflict).
`COLCON_IGNORE` is present so it is skipped by default. To enable:

    rm COLCON_IGNORE
    cp -r . ~/ros2_ws/src/urc_msgs
    cd ~/ros2_ws && colcon build --packages-select urc_msgs urc_rock_detection
    source install/setup.bash
