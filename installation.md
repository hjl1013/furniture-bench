Trouble shooting while installation
```bash
# pip version error
pip install --upgrade pip wheel
pip install setuptools==58
pip install --upgrade pip==22.2.2

# curobo installation
git submodule update --init --recursive # pull curobo in 3dparty
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 # install torch to match cuda toolkit
cd curobo && pip install -e . --no-build-isolation
```