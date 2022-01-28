#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


requirements = (
'numpy', 'torch==1.6.0', 'torchvision==0.7.0', 'opencv-python', 'einops', 'solt==0.1.8', 'tqdm', 'scikit-learn', 'pandas', 'sas7bdat', 'pyyaml', 'matplotlib', 'coloredlogs==14.0', 'hydra-core==1.0.3', 'omegaconf==2.0.2')

setup_requirements = ()

test_requirements = ()

description = """CLIMAT: Clinically-Inspired Multi-Agent Transformers for Knee Osteoarthristic Trajectory Forecasting from Multi-modal Data
"""

setup(
    author="",
    author_email='',
    classifiers=[],
    description="CLIMAT: Clinically-Inspired Multi-Agent Transformers for Knee Osteoarthristic Trajectory Forecasting from Multimodal Data",
    install_requires=requirements,
    license="MIT license",
    long_description=description,
    include_package_data=True,
    keywords='',
    name='CLIMAT',
    packages=find_packages(include=[]),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    version='0.0.1',
    zip_safe=False,
)
