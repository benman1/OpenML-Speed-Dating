#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy', 'pandas', 'scikit-learn', 'category_encoders>=2.0.0', 'numba'
]

setup_requirements = []

test_requirements = []

setup(
    author="Ben Auffarth",
    author_email='auffarth@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="""This contains all the steps needed to
                    reproduce the OpenML pipeline steps
                """,
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='openml_speed_dating_pipeline_steps',
    name='openml_speed_dating_pipeline_steps',
    packages=find_packages(
        include=[
            'openml_speed_dating_pipeline_steps',
            'openml_speed_dating_pipeline_steps.*'
        ]
    ),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/benman1/openml_speed_dating_pipeline_steps',
    version='0.5.6',
    zip_safe=False,
)
