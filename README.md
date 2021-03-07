# more-contrastive

### Requirements

- conda
- Python 3.6 or higher

### Installation

```bash
# If you need to install conda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Clone this repository
git clone https://github.com/ngowilliam1/more-contrastive.git
cd more-contrastive 

# Create env with requirements
conda env create -f environment.yml

# activate the conda environment
conda activate PyConEnv
```

### To Obtain Kitti Dataset
```bash
cd data
# Download Cityscapes annotations and images:
gdown https://drive.google.com/uc?id=1LqqRBDh1LOEsW5NEX2gP5aievmfy0ac4
# Extract
unzip -q KITTI_OD.zip
```

### Download Weights
Download required weights from [PyContrast](https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/docs/MODEL_ZOO.md)



### To train OD
1. Install [Detectron2](https://github.com/facebookresearch/detectron2).
2. Convert pre-trained models to Detectron2 models:
```
python convert_pretrained.py model.pth det_model.pkl
```
3. Set up data folders following Detectron2's [datasets instruction](https://github.com/facebookresearch/detectron2/tree/master/datasets).

4. Go to Detectron2's folder, and run:
```
python tools/train_net.py \
  --num-gpus 8 \
  --config-file /path/to/config/config.yaml \
  MODEL.WEIGHTS /path/to/model/det_model.pkl
```
where `config.yaml` is the config file listed under the [configs](configs) folder.