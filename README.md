
# CSC413 Final Project

## Environment Setup

### Create and Activate Conda Environment
```bash
conda create -n csc413 python=3.8 -y
conda activate csc413
```

### Install PyTorch and Related Packages
```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

### Additional Requirements
```bash
pip install -r requirements.txt
```

### Setup MMDetection3D
```bash
cd /path/to/MapTR_v2_modified/mmdetection3d
python setup.py develop
# Resolve CUDA_HOME not found error if it occurs
conda install -c conda-forge cudatoolkit-dev
```

### Install Geometric Kernel Attention Module
```bash
cd /path/to/MapTR_v2_modified/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install
```

## Map Training and Evaluation

Modify `code/MapTR_v2_modified/mmdetection3d/mmdet3d/apis/test.py` to output results including BEV features in a `.pkl` file with the sample token for each scene.

## Merging Map Data with Trajectory Data

### Installation of Trajdata
```bash
pip install git+https://github.com/NVlabs/trajdata.git
```

### Necessary Files
- `results.pickle` from mapping evaluation
- Trajectory pickle files: `traj_scene_frame_{full_train, full_val, mini_train, mini_val}.pkl`
- Ground truth files: `gt_{full_train, full_val, mini_val}.pickle`

### Data Merging Process
Run the scripts in `/code/adaptor` to merge data ensuring files are in the same scene. For MapTR and MapTR_v2, use the same file `adaptor_maptr.py`

```bash
python adaptor_maptr.py
python adaptor_vis.py  # For visualization
```

## Train Trajectory Prediction Models

### HiVT Setup
Follow its guide and modify `/code/HiVT_modified/models/local_encoder.py` for BEV features integration.

### Training and Testing Scripts
```bash
bash train.sh
bash test.sh
```
## Notes

Thank you for your support!
