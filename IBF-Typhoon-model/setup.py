# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:01:00 2020

@author: ATeklesadik
"""
from setuptools import setup, find_packages
from pathlib import Path

PROJECT_NAME = "typhoonmodel"

setup(
    name=PROJECT_NAME,
    version="0.1",
    author="Aklilu Teklesadik",
    author_email="ateklesadik@redcross.nl",
    description="Typhoon impact forecasting model",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={'': [str(Path(__file__).parent.absolute() /
                       "src/climada/conf/climada.conf")]},  # Needed for climada to work
    entry_points={
        'console_scripts': [
            f"run-typhoon-model = {PROJECT_NAME}.pipeline:main",
        ]
    }
)
