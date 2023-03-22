# CBTNet


## Test Results
Download the test results from [Google Drive]

## Usage
Download the pretrained model from [Google Drive], and place it in the folder `saved_models`. 
Run the following code to generate test results.

Download the test dataset from [Google Drive], and place it in the folder `test_data`. 
Run the following code to generate test results.
```
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