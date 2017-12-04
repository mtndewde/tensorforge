from setuptools import setup


setup(
    name="tforge",
    version="0.1a",
    description="High level constructs for building machine learning models with Tensorflow.",
    author='mtndewde',
    author_email="mtn.dewde@gmail.com",
    packages=["tforge"],
    install_requires=["tensorflow", "h5py"],
    zip_safe=False
)