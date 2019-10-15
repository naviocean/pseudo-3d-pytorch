
# Pseudo-3D Residual Networks

This repo implements the network structure of P3D[1] with PyTorch, pre-trained model weights are converted from caffemodel, which is supported from the [author's repo](https://github.com/ZhaofanQiu/pseudo-3d-residual-networks)


### Requirements:

- pytorch
- numpy
- ffmpeg (for extract image frames from videos)

### Pretrained weights

1, P3D-199 trained on Kinetics dataset:

 [Google Drive url](https://drive.google.com/drive/folders/1u_l-yvhS0shpW6e0tCiqPE7Bd1qQZKdD)
 
2, P3D-199 trianed on Kinetics Optical Flow (TVL1):

 [Google Drive url](https://drive.google.com/drive/folders/1u_l-yvhS0shpW6e0tCiqPE7Bd1qQZKdD)

 
### Prepare Dataset UCF101
First, download the dataset from UCF into the data folder and then extract it.
```
cd data && wget http://crcv.ucf.edu/data/UCF101/UCF101.rar
unrar e UCF101.rar
```

Next, make 3 folders train, test and validation:
```
mkdir train test validation
```
Finally, run scripts to extract image frames from videos;
```
python move.py
python makeVideoFolder.py
python extract.py
```

### Run Code
1, For Training from scratch
```
python main.py /path/data/
```
2, For Fine-tuning
```
python main.py /path/data/ --pretrained
```
3, For Evaluate model
```
python main.py /path/data/ --resume=checkpoint.pth.tar --evaluate
```
4, For testing model
```
python main.py /path/data/ --test
```

### Experiment Result From Us
Dataset | Accuracy
---|---|
UCF-101 | 81.6%
MERL Shopping | 82.6%

Reference:

 [1][Learning Spatio-Temporal Representation with Pseudo-3D Residual,ICCV2017](http://openaccess.thecvf.com/content_iccv_2017/html/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.html)
