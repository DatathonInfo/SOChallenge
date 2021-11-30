#nsml: nvcr.io/nvidia/pytorch:19.07-py3

from distutils.core import setup

setup(
    name='nsml test example',
    version='1.1',
    install_requires=[

        'textdistance==4.2.0',
        'torch==1.10.0',
    ]
)