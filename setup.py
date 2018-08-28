#!/usr/bin/env python

from setuptools import setup, Command, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='FAPSPLMAgents',
      version='0.1.1',
      description='Siemens PLM Machine Learning Agents',
      license='Apache License 2.0',
      author='FAU - FAPS',
      author_email='jupiter.bakakeu@faps.fau.de',
      url='https://faps.de/',
      packages=find_packages(exclude = ['ppo']),
      install_requires = required,
      long_description= ("FAPS PLM Machine Learning Agents allows researchers and developers "
       "to transform games and simulations created using a Siemens Simulation Tool \(i.e NX MCD, PlantSimulation, Process Simulate\) into environments "
       "where intelligent agents can be trained using reinforcement learning, evolutionary " 
       "strategies, or other machine learning methods through a simple to use Python API.")
     )
