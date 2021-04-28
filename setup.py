import os
import setuptools

VERSION = '1.0.0'

folder = os.path.dirname(__file__)
with open(os.path.join(folder, 'requirements.txt')) as fp:
  install_requires = [line.strip() for line in fp]


description = ('NORSE - NOn Recurrent Speech Enhancement')

setuptools.setup(
  name='norse',
  version=VERSION,
  packages=setuptools.find_packages(),
  description=description,
  long_description=description,
  author='Danielius Visockas',
  install_requires=install_requires,
  license='MIT',
)
