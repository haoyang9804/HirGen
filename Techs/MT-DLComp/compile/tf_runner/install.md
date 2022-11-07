# Installation steps for running comparision of glow with tensorflow

Note: 
make sure that the cloned onnx-tensorflow repo lies 
in different path from the one installed by tf2-cpu

```shell
conda create -n tf-glow python=3.9
conda activate tf-glow
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.5.0-cp39-cp39-manylinux2010_x86_64.whl
pip install tqdm onnx
mkdir tf-glow-onnx2tf
cd tf-glow-onnx2tf
git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow
pip install -e .
```