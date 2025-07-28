from setuptools import find_packages
from distutils.core import setup

setup(name='go2-walkthrough',
      version='1.0.0',
      author='Tommy Tang',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='feitongTang@163.com',
      description='Template RL environments for Unitree Robots Go2',
      install_requires=['isaacgym', 'rsl-rl', 'matplotlib', 'numpy==1.24', 'tensorboard', 'mujoco==3.2.3', 'pyyaml'])
