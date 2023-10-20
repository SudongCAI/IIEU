# IIEU
Here is the project code for Instantaneous Importance Estimation Units (from **IIEU: Rethinking Neural Feature Activation from Decision-Making** in ICCV 2023).

### Install Requirements
nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04

pip install --no-cache-dir scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install --no-cache-dir pip==21.0

pip install --no-cache-dir scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install --no-cache-dir scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

pip install --no-cache-dir timm==0.4.12 pyyaml ipdb tqdm einops matplotlib tensorboardx pillow==9.0.1 pycocotools scipy pandas

pip install --no-cache-dir opencv_contrib_python==4.5.2.54

pip install -U openmim

mim install mmcv-full

pip install mmcv-full==1.3.3 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

### Training from scratch
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train_dist.py "$@" <path_to_imagenet> --model iieub_mobilenetv2 --lr 0.5 --warmup-epochs 5 --epochs 240 --sched cosine -b 256 -j 16 --dist-bn reduce

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train_dist_resnet.py "$@" <path_to_imagenet> --model iieub_resnet14 --lr 0.1 --warmup-epochs 5 --epochs 120 --sched cosine -b 64 -j 4 --amp --dist-bn reduce

### Evaluation
python3 validate.py <path_to_imagenet> --model iieub_resnet14 --checkpoint <path_to_checkpoint>


### Cues
* "--model" is replaceable with the corresponding model names.
* The IDs of "CUDA_VISIBLE_DEVICES" are optional and replaceable. Please also set "--nproc_per_node" accordingly.
* "-b" corresponds to the mini-batch for each GPU; As for "train_dist_resnet.py", the total mini-batch and the learning rate are expected to meet "lr = 0.1 * (b * gpu_num) / 256 ." 
* "--amp" is optional.

*Remarks*: 
* The trained model weights can be found at https://pan.baidu.com/s/1KFtOas5rcK-fPnXZcYv5Aw (password: 6b7f) 
* The .xlsx file of the detailed original records of Acc. Eval. (per epoch) and Loss Values Train. are provided (a cleaned version of .csv files, can be found in the above link).
* For reference, the CIFAR-100 code can also be found in the above link.
* Constrained by coding ability, the code may not be optimal w.r.t. memory and speed/runtime.
* Please note that this project code does not include the definition files of Erfact, Pserf, and SMU(s) (potential permission issues may have). If needed, it is recommended to request them from the original author(s).
