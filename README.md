# CBTNet


## Test Results
Download the test results from [Google Drive](https://drive.google.com/drive/folders/1iTgb7ewXZbCxd7E2EAfYhdiJ_7AZURvJ?usp=share_link).

## Get Data and models
Download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/18lWzDn6nlh9TR1oGcDS6GiUcEgn7p91T?usp=share_link), and place it in the folder `saved_models`. 
Run the following code to generate test results.

Download the test dataset from [Google Drive](https://drive.google.com/file/d/1MmTRvJDpqtbhDlzlV2PrO_gqBEIW744o/view?usp=share_link), and place it in the folder `test_data`. 
Run the following code to generate test results.
```
BKchallenge
│   README.md 
│
└───saved_models
│   │   boken_net_g_29500.pth
│   │   deboken_net_g_2000.pth
│   │   DIS_model.pth
│   │   model.pt
│
│   
└───test_data
    │   00000.src.jpg
    │   00001.src.jpg
    │   00002.src.jpg
    |   .... 
```
## Usage
```
cd BKchallenge
sh test.sh
```
## Environment 
you can look for the 'requirements.txt' to see the requirements

for a new environment you can run below codes
```
conda create --name boken_test python==3.8.15

conda activate boken_test

conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch

pip install requirements.txt


```