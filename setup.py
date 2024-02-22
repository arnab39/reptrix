from setuptools import setup, find_packages

setup(
    name='reptrix',  # Replace with your package's name
    version='0.1.0',  # Package version
    author='Arnab Mondal, Arna Ghosh',  # Replace with your name
    author_email='arnab.mondal@mila.quebec, ghosharn@mila.quebec',  # Replace with your email
    description='Library that provides metrics to assess representation quality',  # Package summary
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/arnab39/reptrix',  # Replace with your repository URL
    packages=find_packages(),
    install_requires=[
        'torch',  # Specify your project's dependencies here
        'numpy', 
        'scikit-learn'
        'torchvision',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',  # Minimum version requirement of Python
)