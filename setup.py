from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).parent


REQUIRED = [i.strip() for i in open(HERE / "requirements.txt") if not i.startswith("#")]
setup(
    name="lofiforever",
    version="0.1.0",
    packages=find_packages(),
    install_requires=REQUIRED,
)
