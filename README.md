# DaDL-SChlo
DaDL-SChlo is a predictor of protein subchloroplast localization.

## Datasets
* Datasets/MSchlo578.fasta is the benchmark dataset of multi-label subchloroplast protein sequences.
* Datasets/Novel is the indenpendent test dataset of multi-label subchloroplast protein sequences.
    > Datasets/Novel/new_envelope.fasta is a protein sequence dataset of the envelope location in the Novel dataset.
    > Datasets/Novel/new_stroma.fasta is a protein sequence dataset of the stroma location in the Novel dataset.
    > Datasets/Novel/new_thylakoid_lumen.fasta is a protein sequence dataset of the thylakoid lumen location in the Novel dataset.
    > Datasets/Novel/new_thylakoid_membrane.fasta is a protein sequence dataset of the thylakoid membrane location in the Novel dataset.

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
Users learn deep learning features from ProBERT []

Users learn handcrafted features from 
