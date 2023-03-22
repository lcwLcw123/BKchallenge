mkdir result/
mkdir test_data/
mkdir saved_models/
python IS-Net/Inference.py
python inference_final.py -opt1 options/test/BOKEN/NAFNet-width64_125_debokeh.yml -opt2 options/test/BOKEN/NAFNet-width64_125_bokeh.yml