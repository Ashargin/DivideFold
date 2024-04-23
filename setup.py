from setuptools import setup

setup(
    name="DivideFold",
    description="Secondary structure prediction for long non-coding RNAs using a recursive cutting method",
    author="Loïc Omnes",
    url="https://github.com/Ashargin/DivideFold",
    keywords=[
        "long non-coding RNA",
        "secondary structure prediction",
        "deep learning",
        "divide and conquer approach",
    ],
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=["dividefold"],
    install_requires=[
        "numpy",
        "scipy",
        "keras>=3.2.1",
        "tensorflow",
        "pandas",
    ],
)
