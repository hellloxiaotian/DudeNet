# DudeNet
## Designing and Training of A Dual CNN for Image Denoising 
## This paper is conducted by Chunwei Tian, Yong Xu, Wangmeng Zuo, Bo Du, Chia-wen Lin and David Zhang. It is implemented by Pytorch. And it is reported by Cver at https://wx.zsxq.com/mweb/views/topicdetail/topicdetail.html?topic_id=841142121551482&group_id=142181451122&user_id=28514284588581&from=timeline.

## Absract
### Deep convolutional neural networks (CNNs) for image denoising have recently attracted increasing research interest. However, plain networks cannot recover ﬁne details for a ### complex task, such as real noisy images. In this paper, we propose a Dual denoising Network (DudeNet) to recover a clean image. Speciﬁcally, DudeNet consists of four
### modules: a feature extraction block, an enhancement block, a compression block, and a reconstruction block. The feature extraction block with a sparse mechanism extracts 
### global and local features via two sub-networks. The enhancement block gathers and fuses the global and local features to provide complementary
### informationforthelatternetwork.Thecompressionblockreﬁnes the extracted information and compresses the network. Finally, the reconstruction block is utilized to reconstruct a ### denoised image. The DudeNet has the following advantages: (1) The dual networks with a parse mechanism can extract complementary features to enhance the generalized ability ### of denoiser. (2) Fusing global and local features can extract salient features to recover ﬁne details for complex noisy images. (3) A Small-size ﬁlter is used to reduce the ### complexity of denoiser. Extensive experiments demonstrate the superiority of DudeNet over existing current state-of-the-art denoising methods.

## Requirements (Pytorch)  
#### Pytorch 0.41
#### Python 2.7
#### torchvision 
#### openCv for Python
#### HDF5 for Python



## Commands
### Training
### Training datasets 
#### The  training dataset of the gray noisy images is downloaded at https://pan.baidu.com/s/1nkY-b5_mdzliL7Y7N9JQRQ or https://drive.google.com/open?id=1_miSC9_luoUHSqMG83kqrwYjNoEus6Bj (google drive)
#### The  training dataset of the color noisy images is downloaded at https://pan.baidu.com/s/1ou2mK5JUh-K8iMu8-DMcMw (baiduyun) or https://drive.google.com/open?id=1S1_QrP-fIXeFl5hYY193lr07KyZV8X8r (google drive) 

### Train DuDeNet-S (DuDeNet with known noise level)
#### python train.py --prepropcess True --num_of_layers 17 --mode S --noiseL 25 --val_noiseL 25    

### Train DuDeNet-B (DnCNN with blind noise level)
#### python train.py --preprocess True --num_of_layers 17 --mode B --val_noiseL 25

### Test 
### Gray noisy images
#### python test.py --num_of_layers 17 --logdir g15 --test_data Set68 --test_noiseL 15 
### Gray blind denoising
#### python test_Gb.py --num_of_layers 17 --logdir gblind --test_data Set68 --test_noiseL 25   

### Color noisy images
#### python test_c.py --num_of_layers 17 --logdir g15 --test_data Set68 --test_noiseL 15 
### Color blind denoising
#### python test_c.py --num_of_layers 17 --logdir cblind --test_data Set68 --test_noiseL 15  

### 1. Network architecture
![RUNOOB 图标](./results/fig1.jpg)

### 2. Real noisy images
![RUNOOB 图标](./results/fig2.jpg)

### 3. Effectiveness of key techniques in the DudeNet for image denoising
![RUNOOB 图标](./results/Table1.jpg)

### 4. Run-time of key techniques in the DudeNet for different sizes noisy images
![RUNOOB 图标](./results/Table2.jpg)


### Test Results
#### 5. DudeNet for BSD68
![RUNOOB 图标](./results/Table4.jpg)

#### 6. DudeNet for Set12
![RUNOOB 图标](./results/Table5.jpg)

#### 7. DudeNet for CBSD68 and Kodak24
![RUNOOB 图标](./results/Table6.jpg)

#### 8. DudeNet for real noisy images 
![RUNOOB 图标](./results/Table7.jpg)

#### 9. Run-time of DudeNet for a noisy image of different sizes.
![RUNOOB 图标](./results/Table9.jpg)

### 10.Complexity analysis of different networks.
![RUNOOB 图标](./results/Table3.jpg)

#### 11. Visual results of Set12
![RUNOOB 图标](./results/fig3.jpg)

#### 12. Visual results of BSD68
![RUNOOB 图标](./results/fig4.jpg)

#### 13. Visual results of CBSD68
![RUNOOB 图标](./results/fig5.jpg)

#### 14. Visual results of Kodak24 
![RUNOOB 图标](./results/fig6.jpg)

### If you cite this paper, please refer to the following formats: 
#### 1.Tian C, Xu Y, Zuo W, et al. Designing and Training of A Dual CNN for Image Denoising[J]. arXiv preprint arXiv:2007.03951, 2020.

#### 2. @article{tian2020designing,
####    title={Designing and Training of A Dual CNN for Image Denoising},
####    author={Tian, Chunwei and Xu, Yong and Zuo, Wangmeng and Du, Bo and Lin, Chia-Wen and Zhang, David},
####    journal={arXiv preprint arXiv:2007.03951},
####    year={2020}
####    }
