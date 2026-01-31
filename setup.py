import os
from setuptools import setup, find_packages

setup(
    name="cgmap",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyyaml",
        "panel>=1.0",
        "bokeh>=3.0",
        "networkx",
    ],
    entry_points={
        'console_scripts': [
            'cgmap=cgmap.cgmap:main',
            'cgmap-gui=cgmap.gui.app:_cli_main',
        ],
    },
    author="Zhenghao Wu",
    author_email="w415146142@gmail.com",
    description="A tool for coarse-grain mapping of molecular dynamics trajectories",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cgmap",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 