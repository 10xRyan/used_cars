from setuptools import find_packages,setup
from typing import List

EDOT='-e .'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[i.replace("\n","") for i in requirements]
        
        if EDOT in requirements:
            requirements.remove(EDOT)
    return requirements

setup(
    name='used_car_price_predictor',
    version='1.0',
    author='Ryan_R',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)