# Edge-LBAM
Pytorch implementation of paper "Image Inpainting with Edge-guided Learnable Bidirectional Attention Maps"

## Description

This paper is an extension of our previous work. In comparison to [LBAM](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xie_Image_Inpainting_With_Learnable_Bidirectional_Attention_Maps_ICCV_2019_paper.pdf) we utilize both the mask of holes
and predicted edge map for mask-updating, resulting in our Edge-LBAM method. Moreover, we introduce a multi-scale
edge completion network for effective prediction of coherent edges.

## Prerequisites

- Python 3.6
- Pytorch =1.1.0 
- CPU or NVIDIA GPU + Cuda + Cudnn

## Training


To train the Edge-LBAM model:

```
python train.py --batchSize numOf_batch_size --dataRoot your_image_path \
--maskRoot your_mask_root --modelsSavePath path_to_save_your_model \
--logPath path_to_save_tensorboard_log --pretrain(optional) pretrained_model_path
```

## Testing

To test with random batch with random masks:

```
python test_random_batch.py --dataRoot your_image_path
--maskRoot your_mask_path --batchSize numOf_batch_size --pretrain pretrained_model_path
```

## Pretrained Models

 The pretrained models can be found at [google drive](https://drive.google.com/drive/folders/1iilIU0U7fOYjYlRB7bZjN5oLNCeLoW-R?usp=sharing), we will release the models removing bn from Edge-LBAM later which may effect better. You can also train the model by yourself.

## Results

#### Inpainting
From left to right are input, the result of Global&Local,PConv,DeepFillv2.

<img src="https://github.com/wds1998/Edge-LBAM/blob/main/examples/input28-1.png?raw=true" width="18%"/> <img src="https://github.com/wds1998/Edge-LBAM/blob/main/examples/gl28-1.png?raw=true" width = "18%"/> <img src="https://github.com/wds1998/Edge-LBAM/blob/main/examples/pconv28-1.png?raw=true" width="18%"/><img src="https://github.com/wds1998/Edge-LBAM/blob/main/examples/gc28-1.png?raw=true" width="18%"/>

From left to right are the result of Edge-Connect, MEDFE, Ours and GT.

<img src="https://github.com/wds1998/Edge-LBAM/blob/main/examples/ec28-1.png?raw=true" width="18%"/> <img src="https://github.com/wds1998/Edge-LBAM/blob/main/examples/MEDFE28-1.png?raw=true" width = "18%"/> <img src="https://github.com/wds1998/Edge-LBAM/blob/main/examples/ours28-1.png?raw=true" width="18%"/><img src="https://github.com/wds1998/Edge-LBAM/blob/main/examples/GT28-1.png?raw=true" width="18%"/>    

### MECNet
From left to right are input, edge competion of single-scale network and multi-scale network.

 <img src=examples\input1.png width=30%/><img src=examples\edge_mecnet(s)_1.png width=30%/><img src=examples\edge_mecnet_1.png width=30%/>
