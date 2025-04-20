from setuptools import setup, find_packages

setup(
    name="biosight",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "torch",
        "torchvision",
        "python-multipart",
        "pymongo",
        "prometheus-client",
        "passlib[bcrypt]",
        "python-jose[cryptography]",
        "jinja2"
    ]
)