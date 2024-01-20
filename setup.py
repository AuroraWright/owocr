from pathlib import Path
from setuptools import setup

long_description = (Path(__file__).parent / "README.md").read_text('utf-8').split('# Installation')[0]

setup(
    name="owocr",
    version='0.1',
    description="Japanese OCR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AuroraWright/owocr",
    author="AuroraWright",
    author_email="fallingluma@gmail.com",
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=['owocr'],
    include_package_data=True,
    install_requires=[
        "fire",
        "jaconv",
        "loguru",
        "numpy",
        "Pillow>=10.0.0",
        "pyperclip",
        "pynput",
        "websockets",
        "notify-py"
    ],
    entry_points={
        "console_scripts": [
            "owocr=owocr.__main__:main",
        ]
    },
)
