
## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```
git clone https://github.com/Jian-Nam/DecorDetector-AiEnd.git
cd DecorDetector-AiEnd
pip install -r requirements.txt
```

## Run api server

It will provide two endpoints which is `/segment` and `/vectorize`. `/segment`is for remove background from an image. It works on [Segment-Anything model](https://github.com/facebookresearch/segment-anything) and An image file, pointX, and pointY are required as parameters. `/vectorize` works on Resnet50 model, and it vectorizes an image. An image file is required as parameters. 

Run api server: 

```
python app.py
```