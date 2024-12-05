import torch.nn as nn
from networks.encoder import Encoder
from networks.decoder import Decoder
from networks.DCTlayer import *

class EncoderDecoder(nn.Module):
   
    def __init__(self,encode_message_length = 128,decode_message_length=32,pixel_space=False):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(encode_message_length)
        if not pixel_space:
            self.dctlayer = DCTlayer()
            self.idctlayer = DCTlayer(inverse=True)
        else:
            self.dctlayer = Pixellayer()
            self.idctlayer = Pixellayer(inverse=True)
        self.decoder = Decoder(decode_message_length)

    def forward(self, image, message,random_mask=None):

        img_dct_masked, img_dct, random_mask = self.dctlayer(image,random_mask=random_mask)
        encoded_dct = self.encoder(img_dct_masked, message)
        # The idct layer will use the random mask to filter out unnesscesary value
        # and add back to the img_dct 
        # encoded_dct_origin = encoded_dct*random_mask
        encoded_dct, encoded_image = self.idctlayer(encoded_dct,img_dct,random_mask)

        # img_dct_masked_, img_dct_, random_mask = self.dctlayer(encoded_image,random_mask=random_mask)
        # encoded_dct_, encoded_image_ = self.idctlayer(img_dct_masked_,img_dct,random_mask)
        # print('change:',img_dct_masked_ - encoded_dct_origin)
        decoded_message = self.decoder(encoded_dct)
        # decoded_message_ = self.decoder(encoded_dct_)
        # print('change:',decoded_message - decoded_message_)
        return encoded_image, decoded_message, random_mask,encoded_dct,img_dct_masked.detach()