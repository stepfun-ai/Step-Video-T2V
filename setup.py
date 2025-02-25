from setuptools import find_packages, setup
import subprocess


def get_cuda_version():
    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        version_line = [line for line in nvcc_version.split("\n") if "release" in line][
            0
        ]
        cuda_version = version_line.split(" ")[-2].replace(",", "")
        return "cu" + cuda_version.replace(".", "")
    except Exception as e:
        return "no_cuda"


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    fp = open("stepvideo/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    setup(
        name="stepvideo",
        author="Step-Video Team",
        packages=find_packages(),
        install_requires=[
            "torchvision==0.18",
            "torch==2.3",
            "accelerate>=1.0.0",
            "transformers>=4.39.1",
            "diffusers>=0.31.0",
            "sentencepiece>=0.1.99",
            "imageio>=2.37.0",
            "optimus==2.1",
            "numpy",
            "einops",
            "aiohttp",
            "asyncio",
            "flask",
            "flask_restful",
            "ffmpeg-python",
            "requests",
            "xfuser==0.4.2rc2",
            "nvidia-cublas-cu12==12.6.3.3"
        ],
        url="",
        description="A 30B DiT based text to video and image generation model",
        long_description=long_description,
        long_description_content_type="text/markdown",
        version=version,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        python_requires=">=3.10",
    )