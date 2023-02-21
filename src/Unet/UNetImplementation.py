import torch.nn as nn
import torch.nn.functional as F
import torch
#from skimage.filters import frangi, hessian




class block(nn.Module):
    def __init__(self, chan_in, chan_out):
        super().__init__()
        #Should perform well with random coverlution.
        self.operations = nn.Sequential(
          nn.Conv2d(chan_in,chan_out,kernel_size=3, padding=1),
          nn.BatchNorm2d(chan_out),
          nn.ReLU(),
          nn.Conv2d(chan_out, chan_out,kernel_size=3, padding=1),
          nn.BatchNorm2d(chan_out),
          nn.ReLU()
        )

    def forward(self, input):
        output =self.operations(input)
        return output


class encoder_block(nn.Module):
    def __init__(self,chan_in, chan_out):
        super().__init__()
        self.conv_block =block(chan_in, chan_out)
        self.max_pool =nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        conv_output = self.conv_block.forward(inputs)
        down_output = self.max_pool(conv_output)
        return conv_output, down_output

class decoder_block(nn.Module):
    def __init__(self, chan_in, chan_out):
        super().__init__()
        self.up_conv =  nn.ConvTranspose2d(chan_in, chan_out, kernel_size=2, stride=2)
        self.conv_block= block(chan_out+chan_out, chan_out)

    def forward(self, inputs, enc_out):
        output =self.up_conv(inputs)
        output = torch.cat([output, enc_out], axis=1)
        output = self.conv_block.forward(output)
        return output



class Unet(nn.Module):
    def __init__(self):
        super().__init__()


        #Encoder levels           num_channels
        self.enc_1 =encoder_block(1, 32)
        self.enc_2 =encoder_block(32, 64)
        self.enc_3 = encoder_block(64, 128)

        #Bottom block
        self.bottom= block(128, 256)
        
        #Decoder blocks
        #self.d1 = decoder_block(1028, 512)

        self.dec_1 = decoder_block(256, 128)
        self.dec_2 = decoder_block(128, 64)
        self.dec_3= decoder_block(64, 32)

        #Classifying by 1x1 convolution
        self.outputs = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        #Encoder
        to_dec_1 , down_1 = self.enc_1.forward(inputs) 
        to_dec_2, down_2  =self.enc_2.forward(down_1)
        to_dec_3, down_3 = self.enc_3.forward(down_2)
        
        #Bottom
        b = self.bottom.forward(down_3 ) #b = self.b(p4)
       
        #Decoder
        #d1 = self.d1(b, s4)
        res_1=self.dec_1.forward(b, to_dec_3) 
        res_2 =self.dec_2.forward(res_1, to_dec_2)
        res_3 =self.dec_3.forward(res_2, to_dec_1)

        #Calling classification
        outputs = self.outputs(res_3)
        #To ensure data between 0 and 1
        outputs = nn.Sigmoid()(outputs)
        return outputs