from setuptools import setup, find_packages

setup(
    name="dill_clip",
    version="0.1.0",
    description="Set of tools for running Dill CLIP experiments.",
    packages=find_packages(),
    python_requires=">=3.8",
    dependency_links=["https://download.pytorch.org/whl/cu117"],
    # install_requires=[
    #     "torch==2.0.1+cu117",
    #     "torchaudio==2.0.2+cu117",
    #     "torchvision==0.15.2+cu117",
    #     "wandb==0.15.8",
    #     "transformers[torch] @ git+https://github.com/FoamoftheSea/transformers.git@multiformer-dev",
    #     "evaluate==0.4.0",
    #     "shift_dev @ git+https://github.com/FoamoftheSea/shift-dev.git@bbox-loading",
    #     "torchmetrics==1.2.0",
    #     "jupyterlab",
    #     "ipywidgets",
    #     "numba==0.58.0"
    # ],
)
