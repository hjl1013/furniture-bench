# Trouble shooting
```bash
# pip version error / issac gym error
pip install --upgrade pip wheel
pip install setuptools==58
pip install --upgrade pip==22.2.2
```

# Third Party Packages

## Curobo
```bash
# curobo installation
git submodule update --init --recursive # pull curobo in 3dparty
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 # install torch to match cuda toolkit
cd 3dparty/curobo && pip install -e . --no-build-isolation
```

## FoundationPose
We do not use this at this moment
```bash
# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"

# install dependencies
python -m pip install -r requirements.txt

# Install NVDiffRast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Kaolin (Optional, needed if running model-free setup)
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```