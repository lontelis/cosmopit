import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cosmopit-pkg-pntelis", # Replace with your own username
    version="0.0.1",
    author="Pierros Ntelis",
    author_email="pntelis@cppm.in2p3.fr",
    description="COSMOlogical Python Initial Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lontelis/cosmopit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)