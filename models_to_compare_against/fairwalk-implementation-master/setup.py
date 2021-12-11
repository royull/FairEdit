from distutils.core import setup
import setuptools

setup(
    name='fairwalk',
    packages=['fairwalk'],
    version='0.3.2',
    description='Implementation of the fairwalk algorithm.',
    author='Uriel Singer',
    author_email='urielsinger@cs.technion.ac.il',
    license='MIT',
    url='https://github.com/urielsinger/fairwalk',
    install_requires=[
        'networkx',
        'gensim',
        'numpy',
        'tqdm',
        'joblib>=0.13.2'
    ],
    keywords=['machine learning', 'embeddings'],
)