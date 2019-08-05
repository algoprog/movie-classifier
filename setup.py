from setuptools import setup
from setuptools import find_packages

setup(
    name='movie-classifier',
    version='0.1',
    description='',
    packages=find_packages(),
    install_requires=[
    	'Keras==2.2.4',
	'Keras_Preprocessing==1.0.9',
	'Flask_Cors==3.0.6',
	'Flask==1.0.2',
	'numpy==1.16.3',
	'texttable==1.6.2',
	'scikit_learn==0.21.3'
	]
)
