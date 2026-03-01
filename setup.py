import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'visualizer'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='amr2',
    maintainer_email='amr2@todo.todo',
    description='Record and plot odometry, path, velocity, current, caster angles',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'record_node = visualizer.record:main',
            'plot_node = visualizer.plot:main',
            'caster_kinematic_estimator_node = visualizer.caster_kinematic_estimator:main',
            'key_stop_plot_node = visualizer.key_stop_plot:main',
            'trajectory_triptych = visualizer.trajectory_triptych:main',
            'compare_metrics_table = visualizer.compare_metrics_table:main',
            'pose_error_plot = visualizer.pose_error_plot:main',
        ],
    },
)
