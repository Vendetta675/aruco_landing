from setuptools import setup

package_name = 'takeoff'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],   # VERY IMPORTANT
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'auto_takeoff = takeoff.auto_takeoff:main',
        ],
    },
)
