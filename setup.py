from setuptools import find_packages, setup

setup(
    name = "UKGE-FL",
    version = "0.1.0",
    author = "Mou YongLi, Xiaoyan Jin, Qihui Feng, Gerhard Lakemeyer, Stefan Decker",
    author_email = "mou@dbis.rwth-aachen.de",
    description = ("Enhancing Uncertain Knowledge Graphs Embedding using Fuzzy Logic"),
    license = "MIT",
    url = "https://github.com/mouyongli",
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Topic :: Medical Image Segmentation",
        "License :: OSI Approved :: MIT License",
    ],
)