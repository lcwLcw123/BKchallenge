import os

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from models.CUGAN import CUGAN
from models.CUGAN_stack import CUGAN_stack
from torchvision import transforms
import matplotlib.pyplot as plt
import warnings


class Model_load(nn.Module):
    def __init__(self):
        super(Model_load, self).__init__()
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
        print("CUDA visible devices: " + str(torch.cuda.device_count()))
        print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

        self.model_bokeh_375 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
        self.model_bokeh_625 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
        self.model_bokeh_750 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
        self.model_bokeh_1000 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
       
        self.model_debokeh_375 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
        self.model_debokeh_625 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
        self.model_debokeh_750 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
        self.model_debokeh_1000 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)

        self.model_bokeh_375 = torch.nn.DataParallel(self.model_bokeh_375.cuda())
        self.model_bokeh_625 = torch.nn.DataParallel(self.model_bokeh_625.cuda())
        self.model_bokeh_750 = torch.nn.DataParallel(self.model_bokeh_750.cuda())
        self.model_bokeh_1000 = torch.nn.DataParallel(self.model_bokeh_1000.cuda())


        self.model_debokeh_375 = torch.nn.DataParallel(self.model_debokeh_375.cuda())
        self.model_debokeh_625 = torch.nn.DataParallel(self.model_debokeh_625.cuda())
        self.model_debokeh_750 = torch.nn.DataParallel(self.model_debokeh_750.cuda())
        self.model_debokeh_1000 = torch.nn.DataParallel(self.model_debokeh_1000.cuda())
        

        self.model_stack_debokeh_375 = CUGAN_stack(in_nc=6,out_nc=3,cond_dim=1,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
        self.model_stack_debokeh_625 = CUGAN_stack(in_nc=6,out_nc=3,cond_dim=1,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
        self.model_stack_debokeh_750 = CUGAN_stack(in_nc=6,out_nc=3,cond_dim=1,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
        self.model_stack_debokeh_1000 = CUGAN_stack(in_nc=6,out_nc=3,cond_dim=1,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
        
        self.model_stack_debokeh_375 = torch.nn.DataParallel(self.model_stack_debokeh_375.cuda())
        self.model_stack_debokeh_625 = torch.nn.DataParallel(self.model_stack_debokeh_625.cuda())
        self.model_stack_debokeh_750 = torch.nn.DataParallel(self.model_stack_debokeh_1000.cuda())
        self.model_stack_debokeh_1000 = torch.nn.DataParallel(self.model_stack_debokeh_1000.cuda())
        
        
        self.checkpoint = torch.load("./saved_models/model.pt")


#-------------------------------------------------------------------------------------------------------------------

        self.model_bokeh_375.load_state_dict(self.checkpoint['model_path_bokeh_375'],strict=True)
        self.model_bokeh_625.load_state_dict(self.checkpoint['model_path_bokeh_625'],strict=True)
        self.model_bokeh_750.load_state_dict(self.checkpoint['model_path_bokeh_750'],strict=True)
        self.model_bokeh_1000.load_state_dict(self.checkpoint['model_path_bokeh_1000'],strict=True)

#-----------------------------------------------------------------------------------------------------------

        self.model_debokeh_375.load_state_dict(self.checkpoint['model_path_debokeh_375'],strict=True)
        self.model_debokeh_625.load_state_dict(self.checkpoint['model_path_debokeh_625'],strict=True)
        self.model_debokeh_750.load_state_dict(self.checkpoint['model_path_debokeh_750'],strict=True)
        self.model_debokeh_1000.load_state_dict(self.checkpoint['model_path_debokeh_1000'],strict=True)
        print('already load model')

        self.model_bokeh_375.eval()
        self.model_bokeh_625.eval()
        self.model_bokeh_750.eval()
        self.model_bokeh_1000.eval()

        self.model_debokeh_375.eval()
        self.model_debokeh_625.eval()
        self.model_debokeh_750.eval()
        self.model_debokeh_1000.eval()
#--------------------------------------------------------------------------------------------------------
        self.model_stack_debokeh_375.load_state_dict(self.checkpoint['model_path_stack_debokeh_375'],strict=True)
        self.model_stack_debokeh_625.load_state_dict(self.checkpoint['model_path_stack_debokeh_625'],strict=True)
        self.model_stack_debokeh_750.load_state_dict(self.checkpoint['model_path_stack_debokeh_750'],strict=True)
        self.model_stack_debokeh_1000.load_state_dict(self.checkpoint['model_path_stack_debokeh_1000'],strict=True)


        self.model_stack_debokeh_375.eval()
        self.model_stack_debokeh_625.eval()
        self.model_stack_debokeh_750.eval()
        self.model_stack_debokeh_1000.eval()
#---------------------------------------------------------------------------------------------------------
    def inference(self,batch):

        source = batch["source"].cuda()
        source_alpha = batch["source_alpha"].cuda()

        source_output = source.clone()
        source = source*(1-source_alpha)

        input_cond = batch["input_cond"].cuda()
        output_cond = batch["output_cond"].cuda()
        
        extent = output_cond[0][0].item()
   
        with torch.no_grad():
            if batch['tgt_blur'] == True:
                if extent == 0.125 or extent == 0.25:
                    output = source_output
                elif extent == 0.375:
                    output = self.model_bokeh_375(source,output_cond,output_cond)
                    output = output*(1-source_alpha)+source_output*source_alpha
                    output = torch.clamp(output,min=0.0, max=1.0)


                elif extent == 0.625:
                    output = self.model_bokeh_625(source,output_cond,output_cond)
                    output = output*(1-source_alpha)+source_output*source_alpha
                    output = torch.clamp(output,min=0.0, max=1.0)

                elif extent == 0.75:
                    output = self.model_bokeh_750(source,output_cond,output_cond)
                    output = output*(1-source_alpha)+source_output*source_alpha
                    output = torch.clamp(output,min=0.0, max=1.0)
                    
                elif extent == 1.0:
                    output = self.model_bokeh_1000(source,output_cond,output_cond)
                    output = output*(1-source_alpha)+source_output*source_alpha
                    output = torch.clamp(output,min=0.0, max=1.0)

            elif batch['tgt_blur'] == False:
                if extent == 0.125 or extent == 0.25:
                    output = source_output
                elif extent == 0.375:
                    # output = self.model_debokeh_375(source,output_cond,output_cond)
                    # output = output*(1-source_alpha)+source_output*source_alpha
                    # output = torch.clamp(output,min=0.0, max=1.0)

                    output1 = self.model_debokeh_375(source,output_cond,output_cond)
                    output1 = output1*(1-source_alpha)
                    #`output1 = torch.clamp(output1,min=0.0, max=1.0)
                    output1 = torch.cat((output1, source), 1)
                    disparity = batch["disparity"].cuda()
                    output1 = self.model_stack_debokeh_375(output1,disparity,disparity)
                    output1 = output1*(1-source_alpha)+source_output*source_alpha
                    output = torch.clamp(output1,min=0.0, max=1.0)

                elif extent == 0.625:
                    # output = self.model_debokeh_625(source,output_cond,output_cond)
                    # output = output*(1-source_alpha)+source_output*source_alpha
                    # output = torch.clamp(output,min=0.0, max=1.0)

                    output1 = self.model_debokeh_625(source,output_cond,output_cond)
                    output1 = output1*(1-source_alpha)
                    #output1 = torch.clamp(output1,min=0.0, max=1.0)
                    output1 = torch.cat((output1, source), 1)
                    disparity = batch["disparity"].cuda()
                    output1 = self.model_stack_debokeh_625(output1,disparity,disparity)
                    output1 = output1*(1-source_alpha)+source_output*source_alpha
                    output = torch.clamp(output1,min=0.0, max=1.0)

                elif extent == 0.75:
                    # output = self.model_debokeh_750(source,output_cond,output_cond)
                    # output = output*(1-source_alpha)+source_output*source_alpha
                    # output = torch.clamp(output,min=0.0, max=1.0)

                    output1 = self.model_debokeh_750(source,output_cond,output_cond)
                    output1 = output1*(1-source_alpha)
                    #output1 = torch.clamp(output1,min=0.0, max=1.0)
                    output1 = torch.cat((output1, source), 1)
                    disparity = batch["disparity"].cuda()
                    output1 = self.model_stack_debokeh_750(output1,disparity,disparity)
                    output1 = output1*(1-source_alpha)+source_output*source_alpha
                    output = torch.clamp(output1,min=0.0, max=1.0)

                elif extent == 1.0:
                    # output = self.model_debokeh_1000(source,output_cond,output_cond)
                    # output = output*(1-source_alpha)+source_output*source_alpha
                    # output = torch.clamp(output,min=0.0, max=1.0)

                    output1 = self.model_debokeh_1000(source,output_cond,output_cond)
                    output1 = output1*(1-source_alpha)
                    #output1 = torch.clamp(output1,min=0.0, max=1.0)
                    output1 = torch.cat((output1, source), 1)
                    disparity = batch["disparity"].cuda()
                    output1 = self.model_stack_debokeh_1000(output1,disparity,disparity)
                    output1 = output1*(1-source_alpha)+source_output*source_alpha
                    output = torch.clamp(output1,min=0.0, max=1.0)
                    
        return output

torch.backends.cudnn.deterministic = True
device = torch.device("cuda")
print("CUDA visible devices: " + str(torch.cuda.device_count()))
print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

model_bokeh_375 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
model_bokeh_625 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
model_bokeh_750 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
model_bokeh_1000 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
model_debokeh_375 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
model_debokeh_625 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
model_debokeh_750 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
model_debokeh_1000 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)

model_bokeh_375 = torch.nn.DataParallel(model_bokeh_375.cuda())
model_bokeh_625 = torch.nn.DataParallel(model_bokeh_625.cuda())
model_bokeh_750 = torch.nn.DataParallel(model_bokeh_750.cuda())
model_bokeh_1000 = torch.nn.DataParallel(model_bokeh_1000.cuda())
model_debokeh_375 = torch.nn.DataParallel(model_debokeh_375.cuda())
model_debokeh_625 = torch.nn.DataParallel(model_debokeh_625.cuda())
model_debokeh_750 = torch.nn.DataParallel(model_debokeh_750.cuda())
model_debokeh_1000 = torch.nn.DataParallel(model_debokeh_1000.cuda())

model_stack_debokeh_375 = CUGAN_stack(in_nc=6,out_nc=3,cond_dim=1,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
model_stack_debokeh_625 = CUGAN_stack(in_nc=6,out_nc=3,cond_dim=1,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
model_stack_debokeh_750 = CUGAN_stack(in_nc=6,out_nc=3,cond_dim=1,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
model_stack_debokeh_1000 = CUGAN_stack(in_nc=6,out_nc=3,cond_dim=1,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)

model_stack_debokeh_375 = torch.nn.DataParallel(model_stack_debokeh_375.cuda())
model_stack_debokeh_625 = torch.nn.DataParallel(model_stack_debokeh_625.cuda())
model_stack_debokeh_750 = torch.nn.DataParallel(model_stack_debokeh_750.cuda())
model_stack_debokeh_1000 = torch.nn.DataParallel(model_stack_debokeh_1000.cuda())


model_path_bokeh_750 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_mse/bokeh128_mse_0.75_3_18_epoch10.pth"
model_path_bokeh_375 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_mse/bokeh128_mse_0.375_3_18_epoch12.pth"
model_path_bokeh_625 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_mse/bokeh128_mse_0.625_3_18_epoch2.pth"
model_path_bokeh_1000 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_mse/bokeh128_mse_1.0_3_21_epoch0.pth"
#----------------------------------------------------------------------------------------------------------------------------------
model_path_debokeh_375 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_finetune/debokeh128_preign_l1_0.375_3_9_epoch17.pth"
model_path_debokeh_625 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_finetune/debokeh128_preign_l1_0.625_3_8_epoch22.pth"
model_path_debokeh_750 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_finetune/debokeh128_preign_l1_0.75_3_9_epoch18.pth"
model_path_debokeh_1000 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_finetune/debokeh128_preign_l1_1.0_3_9_epoch13.pth"


model_path_stack_debokeh_375 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_stackfinetune/debokeh128_stack_0.375_3_10_epoch20.pth"
model_path_stack_debokeh_625 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_stackfinetune/debokeh128_stack_0.625_3_10_epoch22.pth"
model_path_stack_debokeh_750 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_stackfinetune/debokeh128_stack_0.75_3_10_epoch10.pth"
model_path_stack_debokeh_1000 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_stackfinetune/debokeh128_stack_1.0_3_10_epoch16.pth"
# model_path_stack_debokeh_375 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_stackfinetune/debokeh128_stackmse_0.375_3_21_epoch0.pth"
# model_path_stack_debokeh_625 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_stackfinetune/debokeh128_stackmse_0.625_3_21_epoch0.pth"
# model_path_stack_debokeh_750 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_stackfinetune/debokeh128_stackmse_0.75_3_21_epoch0.pth"
# model_path_stack_debokeh_1000 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_stackfinetune/debokeh128_stackmse_1.0_3_21_epoch0.pth"

# PATH = "model.pt"
# # model = Model() 
# torch.save({
#             'model_path_bokeh_375': torch.load(model_path_bokeh_375),
#             'model_path_bokeh_625': torch.load(model_path_bokeh_625),
#             'model_path_bokeh_750': torch.load(model_path_bokeh_750),
#             'model_path_bokeh_1000': torch.load(model_path_bokeh_1000),
#             'model_path_debokeh_375': torch.load(model_path_debokeh_375),
#             'model_path_debokeh_625': torch.load(model_path_debokeh_625),
#             'model_path_debokeh_750': torch.load(model_path_debokeh_750),
#             'model_path_debokeh_1000': torch.load(model_path_debokeh_1000),
#             'model_path_stack_debokeh_375': torch.load(model_path_stack_debokeh_375),
#             'model_path_stack_debokeh_625': torch.load(model_path_stack_debokeh_625),
#             'model_path_stack_debokeh_750': torch.load(model_path_stack_debokeh_750),
#             'model_path_stack_debokeh_1000': torch.load(model_path_stack_debokeh_1000),
#             }, PATH)

# print('save ok!')
# model2 = Model_load()