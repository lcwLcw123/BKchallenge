import glob
import os.path as osp
# from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options1,parse_options2
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite


from torchvision.transforms import ToPILImage, ToTensor
import cv2
import os
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from dataset.dataset_final import BokehDataset
from models.CUGAN import CUGAN
from models.CUGAN_stack import CUGAN_stack
from torchvision import transforms
import matplotlib.pyplot as plt
import warnings
from Model_final import Model_load
warnings.filterwarnings("ignore")
to_image = transforms.Compose([transforms.ToPILImage()])
to_tensor = ToTensor()
to_pil = ToPILImage()

np.random.seed(0)
torch.manual_seed(0)

def train():
    
    model = Model_load()
    model.eval()
    
    test_dataset = BokehDataset("./test_data", transform=ToTensor())
    #test_dataset = BokehDataset("/home2/chenzigeng/Bokeh_v7/val", transform=ToTensor())
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True,num_workers=2)

    test_iter = iter(test_dataloader)
    total_num = 1*len(test_iter)
    

    opt1 = parse_options1(is_train=False)
    opt1['num_gpu'] = torch.cuda.device_count()
    file_client = FileClient('disk')
    opt1['dist'] = False
    model_debokeh = create_model(opt1)

    opt2 = parse_options2(is_train=False)
    opt2['num_gpu'] = torch.cuda.device_count()
    file_client = FileClient('disk')
    opt2['dist'] = False
    model_bokeh = create_model(opt2)




    print("total_num:",total_num)
 
    for i in tqdm(range(total_num)):

        batch = next(test_iter)
        
        output_cond = batch["output_cond"].cuda()
        
        extent = output_cond[0][0].item()
        disparity = batch["disparity"].cuda()
        id = batch["id"]
        print(id,output_cond)

        if extent >0.3:
            #continue
            output = model.inference(batch)
        
        # else:
        #     if disparity>0.5:
        #         #continue
        #         output, target = model.inference(batch)
        #     elif batch['tgt_blur'] == True:
                
        #         source = batch["source"].cuda()
                
        #         source_origin = source.clone()
                
        #         source_alpha = batch["source_alpha"].cuda()
        #         source = source*(1-source_alpha)
                
        #         source = np.array(to_image(torch.squeeze(source.float().detach().cpu())))
        #         source = source[:,:,(2,1,0)]
        #         cv2.imwrite ( f"src.jpg", source)


        #         path ="src.jpg"
                
        #         img_bytes = file_client.get(path, None)
                
                
        #         try:
        #             img = imfrombytes(img_bytes, float32=True)
                    
                    
        #         except:
        #             raise Exception("path {} not working".format(img_path))

        #         img = img2tensor(img, bgr2rgb=True, float32=True)
                


        #         model_bokeh.feed_data(data={'lq': img.unsqueeze(dim=0)})
        #         model_bokeh.test()
        #         visuals = model_bokeh.get_current_visuals()
        #         result = visuals['result']
        #         result = result.cuda() + source_origin*source_alpha
        #         output = torch.clamp(result,min=0.0, max=1.0)
                



        #     elif batch['tgt_blur'] == False:

        #         source = batch["source"].cuda()

        #         source_origin = source.clone()

        #         source_alpha = batch["source_alpha"].cuda()
        #         source = source*(1-source_alpha)
                
            
        #         source = np.array(to_image(torch.squeeze(source.float().detach().cpu())))
        #         source = source[:,:,(2,1,0)]
        #         cv2.imwrite ("src.jpg", source)

        #         path ="src.jpg"
                
        #         img_bytes = file_client.get(path, None)
                
                
        #         try:
        #             img = imfrombytes(img_bytes, float32=True)
                    
        #         except:
        #             raise Exception("path {} not working".format(img_path))

        #         img = img2tensor(img, bgr2rgb=True, float32=True)

        #         model_debokeh.feed_data(data={'lq': img.unsqueeze(dim=0)})
        #         model_debokeh.test()
        #         visuals = model_debokeh.get_current_visuals()
        #         result = visuals['result']
        #         result = result.cuda() + source_origin*source_alpha
        #         output = torch.clamp(result,min=0.0, max=1.0)
        

            output = np.array(to_image(torch.squeeze(output.float().detach().cpu())))
            output = output[:,:,(2,1,0)]
            cv2.imwrite("./result/"+id[0]+".src.jpg", output)
        


    
    #print(f"Metrics: lpips={lpips:0.05f}, psnr={psnr:0.05f}, ssim={ssim:0.05f}")



if __name__ == "__main__":
    train()