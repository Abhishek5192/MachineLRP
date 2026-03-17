from setuptools import setup,find_packages

HAS_E_IN_REQUIREMENTS = '-e .' 

def get_requirements(file_path):
    with open(file_path) as f:
        requirements = f.read().splitlines()

    if HAS_E_IN_REQUIREMENTS in requirements:
        requirements.remove(HAS_E_IN_REQUIREMENTS)
    
    return requirements

setup(
    name='MACHINE-LEARNING',
    version='0.0.1',
    author='Abhishek Gupta',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)