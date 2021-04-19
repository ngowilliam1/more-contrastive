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
conda create -n PyConEnv python=3.7 -y
conda activate PyConEnv
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt

```

### To Obtain Kitti Dataset
```bash
cd detectron2/datasets
# Download Kitti annotations and images:
gdown https://drive.google.com/uc?id=1wQyPZfdtgQ9g2UTErkr1W1f-aUk9gpgm
# Extract
tar -xvf KITTI_OD.tar.xz
```

### To Obtain nuImages Dataset

Download nuImages samples from nuScenes [website](https://www.nuscenes.org/download)

### Download Weights
Download the pretrained weight:
| Model |
|-|
|[InfoMin200](https://drive.google.com/file/d/10kHOwtlzufZI8wCtqycfQm4EXGlGfrZk/view?usp=sharing)|
|[InfoMin800](https://drive.google.com/file/d/1kOKXSfs_7zJkn4tEhgAUDpjxsXDaqGM3/view?usp=sharing) |
|[InfoMin800v2](https://drive.google.com/file/d/1VBCo5B4Zlc0G2VkA2Wl7qMTwo013wsvM/view?usp=sharing) |

(These weights are originally from [PyContrast](https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/docs/MODEL_ZOO.md) but the format is slightly changed to allow retraining and some inconsistencies in the layers' names were fixed).

### To Contrastively Retrain InfoMin
```
python main_contrast.py --method InfoMin --cosine --data_folder path/to/dataset/ --multiprocessing-distributed --world-size 1 --rank 0 -j 4 --batch_size 32 --resume path/to/pretrained/weights.pth --learning_rate 0.003 --epochs 100
```

### To train OD
1. Install Detectron2
```
cd detectron2
python -m pip install ./
```
2. Convert pre-trained models to Detectron2 models:
```
python convert_pretrained.py model.pth det_model.pkl
```
3. Go to Detectron2's folder, and run:
```
python tools/train_net.py \
  --num-gpus 1 \
  --config-file /path/to/config/config.yaml \
  MODEL.WEIGHTS /path/to/model/det_model.pkl
```
where `config.yaml` is the config file listed under the [configs](detectron2/configs/kitti) folder.
