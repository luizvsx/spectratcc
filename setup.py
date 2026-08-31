from setuptools import find_packages, setup

setup(
    name="tcc_project",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "spectra=spectra.pipeline:cli",
            "spectra-web=spectra.web:run",
        ]
    },
)
