import torch
from transformers import GPT2LMHeadModel
from transformers import ViltModel
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForPreTraining,ViltProcessor
import numpy as np
vilprocessor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
tokenizer = vilprocessor.tokenizer

class Evlgpt(nn.Module):
    def __init__(self):
        super(Evlgpt, self).__init__()
        self.vision_model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")
        self.fusion_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        #build gpt decoder model
        self.decoder_model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.pad_token_id)
    
    def vision_forward(self,pixel_values,bool_masked_pos):
        # vision encoding
        
        outputs = self.vision_model(pixel_values, bool_masked_pos=bool_masked_pos,output_hidden_states=True)
        vision_embs=outputs.hidden_states[0]
        
        return vision_embs,outputs.loss

    def fusion_forward(self,input_ids,vision_embs,attention_mask,padded_output_ids):
        # multi-modality fusion
        pixel_mask = torch.ones(vision_embs.size()[:-1])
        fusion_feature = self.fusion_model(input_ids=input_ids,image_embeds=vision_embs,attention_mask=attention_mask,pixel_mask=pixel_mask,output_hidden_states=True).last_hidden_state 
        language_output = self.decoder_model(inputs_embeds=fusion_feature,labels=padded_output_ids)
        return language_output.loss


    def forward(self,pixel_values,bool_masked_pos,input_ids,attention_mask,padded_output_ids):
        # pixel_values,bool_masked_pos,input_ids,attention_mask,padded_output_ids=data
        video_mae_outputs = self.vision_model(pixel_values, bool_masked_pos=bool_masked_pos,output_hidden_states=True)
        vision_embs = video_mae_outputs.hidden_states[0]
        vision_loss = video_mae_outputs.loss
        pixel_mask = torch.ones(vision_embs.size()[:-1])
        fusion_feature = self.fusion_model(input_ids=input_ids,image_embeds=vision_embs,attention_mask=attention_mask,pixel_mask=pixel_mask,output_hidden_states=True).last_hidden_state 
        language_output = self.decoder_model(inputs_embeds=fusion_feature,labels=padded_output_ids)
        return vision_loss+0.1*language_output.loss



    

if __name__=="__main__":
    model = Evlgpt()
    data=model.get_data()
    vision_loss,language_loss=model.forward(data)
    model.update_net(vision_loss,language_loss)
    

    





























