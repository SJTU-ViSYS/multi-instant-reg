# Multi-instance-Point-Cloud-Registration-by-Efficient-Correspondence-Clustering
This is the source code of our CVPR paper [Arxiv](https://arxiv.org/abs/2111.14582)

## Install Environment:
`conda env create -f multiregister.yaml`

## Test on synthetic dataset and real dataset (Scan2CAD)
All the experimental code files are in `./synthetic&real`

### Weights
Download the weights and put them into `./synthetic&real/snapshot`

### Datas
The Scan2CAD dataset may need to be downloaded on [Scan2CAD](https://github.com/skanti/Scan2CAD), we would upload a processed version for testing soon.

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
Download the datas and put them into `./rgbd`
### Run
`sh run.sh` to run and you can see the visualized results.
