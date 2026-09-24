#!/usr/bin/env bash
# One-time Dev 2 setup: ROS 2 Lyrical + RTAB-Map + Gazebo Jetty (ros_gz) on WSL2 Ubuntu 26.04.
# Run in YOUR WSL terminal (needs sudo password):
#   wsl -d Ubuntu-26.04
#   bash /mnt/n/coding/projects/whoami/ugv_nav/docs/localization/setup_wsl_lyrical.sh
set -euo pipefail

. /etc/os-release
if [[ "${VERSION_ID}" != "26.04" ]]; then
  echo "expected Ubuntu 26.04 (Lyrical Tier 1), got ${VERSION_ID}" >&2
  exit 1
fi

sudo apt update
sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe

# ROS 2 apt source (official ros-apt-source package)
if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
  ver=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
        | grep -F '"tag_name"' | awk -F'"' '{print $4}')
  curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ver}/ros2-apt-source_${ver}.${UBUNTU_CODENAME:-${VERSION_CODENAME}}_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb
fi

sudo apt update
sudo apt install -y \
  ros-lyrical-desktop \
  ros-lyrical-rtabmap-ros \
  ros-lyrical-ros-gz \
  ros-lyrical-tf2-tools \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-pytest \
  python3-numpy \
  python3-yaml

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  sudo rosdep init
fi
rosdep update

grep -q "/opt/ros/lyrical/setup.bash" ~/.bashrc || echo "source /opt/ros/lyrical/setup.bash" >> ~/.bashrc

echo
echo "Installed. Versions:"
source /opt/ros/lyrical/setup.bash
dpkg -s ros-lyrical-rtabmap-ros | grep -E '^Version'
dpkg -s ros-lyrical-ros-gz | grep -E '^Version'
gz sim --version 2>/dev/null | head -1 || true
