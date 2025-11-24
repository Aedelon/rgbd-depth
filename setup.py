#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_packages, setup

setup(
    name="rgbddepth",
    version="1.0.0",
    packages=find_packages(),
    author="Manipulation as in Simulation Suite Contributors",
    description="Camera Depth Models for sim-to-real depth transfer",
    long_description="CDM produces clean, simulation-like depth maps from noisy real-world camera data, enabling policies trained purely in simulation to transfer directly to real robots.",
    url="https://github.com/your-org/manip-as-in-sim-suite",
    license="Apache-2.0",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "pillow>=8.0.0",
        "matplotlib",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache-2.0",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
