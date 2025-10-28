from setuptools import find_packages, setup
from glob import glob

package_name = 'xwr_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/launch', glob('launch/*')),
    ],
    install_requires=['setuptools', 'xwr'],
    zip_safe=True,
    maintainer='Ray',
    maintainer_email='juiteh@andrew.cmu.edu',
    description='A ROS2 wrapper for the XWR radar streaming library',
    entry_points={
        'console_scripts': [
            'stream = xwr_ros.stream:main',
            'process = xwr_ros.process:main',
        ],
    },
)
