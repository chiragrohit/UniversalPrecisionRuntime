from setuptools import setup, find_packages

setup(
    name="upr",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "tqdm>=4.65.0",
        "huggingface_hub>=0.19.0"
    ]
)
