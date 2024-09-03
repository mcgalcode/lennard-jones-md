from setuptools import setup

setup(name='lennard_jones_md',
      version='0.1',
      description='A simple Lennard Jones potential',
      url='http://github.com/storborg/funniest',
      author='Max Gallant',
      author_email='maxg@lbl.gov',
      license='MIT',
      packages=['lennard_jones_md'],
      zip_safe=False,
      install_requires=[
          'numpy',
          'scipy'
      ])