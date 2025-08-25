from numpy import insert
from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]


setup(
    name="roadsafe-ai-detect-locate-notify",
    version="0.1.0",
    description="road damage detection and authority notification for safer urban mobility",
    #license=" ",
    #author=" ",
    #author_email=" ",
    #url=" ",
    install_requires=requirements,
    packages=find_packages(),
    #test_suite="tests",
    # include_package_data: to install data from MANIFEST.in
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "roadsafe=roadsafe.cli:main",
        ],
    },
)
