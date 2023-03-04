# DaDL-SChlo

#### DaDL-SChlo is a predictor of protein subchloroplast localization. 
It includes three steps: feature extraction, data augmentation and multi-location prediction.
First, based on protein sequences, the deep learning features extracted by pre-trained model ProtBERT are fused with handcrafted features of PositionSpecific Score Matrix with Automatic Cross-Covariance (ACC-PSSM) to construct a fusion feature set. Second, utilizing the fusion feature set trains Wasserstein GAN with gradient penalty (WGAN-gp) to synthesize feature samples and supplement the original fusion feature set. Third, a hybrid neural network of transformer encoder and convolutional neural network is used to predict subchloroplast locations based on the extended feature set after supplementation.

![](https://github.com/xwanggroup/DaDL-SChlo/blob/master/DaDL-SChlo.jpg)

## data
* **MSchlo578.txt** is the benchmark dataset of multi-label subchloroplast protein sequences.
* **Novel** is the indenpendent test dataset of multi-label subchloroplast protein sequences.

    > **new_envelope.fasta** is a protein sequence dataset of the envelope location in the Novel dataset.
    > 
    > **new_stroma.fasta** is a protein sequence dataset of the stroma location in the Novel dataset.
    > 
    > **new_thylakoid_lumen.fasta** is a protein sequence dataset of the thylakoid lumen location in the Novel dataset.
    > 
    > **new_thylakoid_membrane.fasta** is a protein sequence dataset of the thylakoid membrane location in the Novel dataset.

## Code
### Environment requirement
The code has been tested running under Python 3.7.8. The required packages are as follows:
* numpy == 1.18.1
* pandas == 1.0.1
* torch == 1.7.1
* scikit-learn == 0.24.2
* NVIDIA-SMI == 465.19.01    
* Driver Version == 465.19.01    
* CUDA Version == 11.3   

### Usage 
* The python script **load_data.py** is the implementation of load protein sequences.  
* **protbert_feature.py** is for the deep learning feature extraction by the pre-trained protein language model ProtBert. 
* **feature_selection.py** is for feature selection by XGBoost method. 
* **gan_load.py** is loading the required data augmentation dataset. 
* **gan_train.py** is for training feature data augmentation model. 
* **DaDL-SChlo.py** is used to train the final DaDL-SChlo model for independent testing.
* Folder ‘model’ contains the final model file **Model.pth** for the proposed predictor DaDL-SChlo.


### Notes

Users learn deep learning features from ProtBERT [https://github.com/agemagician/ProtTrans/blob/master/Embedding/PyTorch/Basic/ProtBert.ipynb]   

For handcrafted feature ACC-PSSM, users can extracted it through the website [http://bioinformatics.hitsz.edu.cn/Pse-in-One2.0/PROTEIN/ACC-PSSM] 
or from ncbi-blast [https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/] 

Users also can use their own data to train this data augmentation modules.

