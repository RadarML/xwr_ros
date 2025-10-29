# xwr_ros
A ROS2 wrapper for XWR

depends on uv and [colcon-uv](https://github.com/nzlz/colcon-uv)
```
pip install uv colcon-uv
```

will create a venv when the package is build with colcon build
```
export UV_VENV_CLEAR=1
export UV_LINK_MODE=copy
colcon build
```