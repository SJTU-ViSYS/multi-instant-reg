# Multi-instance-Point-Cloud-Registration-by-Efficient-Correspondence-Clustering
This is the source code of our CVPR paper [Arxiv](https://arxiv.org/abs/2111.14582)

## Install Environment:
`conda env create -f multiregister.yaml`

## Test on synthetic dataset and real dataset (Scan2CAD, ModelNet40)
All the experimental code files are in `./synthetic&real`

### Weights
Download the [weights](https://sjtueducn-my.sharepoint.com/:f:/g/personal/weixuantang_sjtu_edu_cn/EqN_-RBECS5FgQC8F7Ult1wBzSpUu8qj4_sfHG7u8zTikw?e=dbaL51) and put `multi_oneTomore_multi_1` and `multi_real_box_test_main_cad` directly into `./synthetic&real/snapshot`

### Datas
The Scan2CAD dataset may need to be downloaded on [Scan2CAD](https://github.com/skanti/Scan2CAD), and put `./split.json` in the dataset folder. 
If you choose ModelNet40 for synthetic experiments, then you may download the dataset in [Data provided by Pointnet++](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and put `./modelnet40_train.json` and `./modelnet40_test.json` into the dataset folder. 

### Install pytorch extension:
```
cd ./synthetic&real 
cd cpp_wrappers 
sh build.sh 
cd ..
```
### Run
There are two commands to conduct the two experiments respectively in `run.sh`, you can choose which to run.

## Test on our rgbd data
All the experimental code files are in `./rgbd`

### Datas
Download the [datas](https://sjtueducn-my.sharepoint.com/:f:/g/personal/weixuantang_sjtu_edu_cn/Euun43F7Ma1DrrKGtS9Q_CUBnO6ardmpksB3ZJnxMa_YnQ?e=2Ipbwy) and put `scenes` and `objects` directly into `./rgbd`

### Run
`sh run.sh` to run and you can see the visualized results.
