#used to build our application as a package
from setuptools import find_packages,setup
from typing import List

def get_requirements(path:str)->List[str]:
    """
    this function will return requirements in the form of list
    """
    requirements = []
    with open(path) as r:
        requirements = r.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name = 'MLProject',
    version='0.0.1',
    author='Adhityan',
    author_email='Adhityanmudlayar27@gmail.com',
    packages = find_packages(),
    install_requires=get_requirements('requirements.txt')
)