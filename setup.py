from setuptools import find_packages, setup

setup(
    name="deepex",
    version="0.0.1",
    author="Chenguang Wang, Xiao Liu, Zui Chen, Haoyun Hong, Jie Tang, Dawn Song",
    author_email="25714264+cgraywang@users.noreply.github.com",
    description="Zero-Shot Information Extraction as a Unified Text-to-Triple Translation",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP deep learning zero-shot information extraction",
    license="Apache",
    url="https://github.com/cgraywang/deepex",
    package_dir={"": "src"},
    packages=find_packages("src"),
    setup_requires=[
        'setuptools>=18.0',
    ],
    python_requires=">=3.7.0",
    classifiers=[
        "Development Status :: 0",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
    ],
)