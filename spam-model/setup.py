from setuptools import setup

setup(
    name='Spam Model',
    url='https://github.com/jladan/package_demo',
    author='Jacob Patton',
    author_email='jacobpattondev@outlook.com',
    # Needed to actually package something
    packages=['spam-model'],
    # Needed for dependencies
    install_requires=['numpy', 'torch', 'pandas', 'sklearn', 'nltk'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
)