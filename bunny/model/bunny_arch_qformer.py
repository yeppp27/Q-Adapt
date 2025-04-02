from abc import ABC, abstractmethod

import torch

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertLMHeadModel as BertLMHeadModelRaw
from transformers import CLIPTextModel, CLIPTextConfig, AutoTokenizer,CLIPTextModelWithProjection
from .qformer import BertConfig
from .qformer import BertLMHeadModel as BertLMHeadModelQF




from bunny.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
import torch.nn as nn

local_rank=None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)




class BunnyQformer_MetaModel:

    def __init__(self, config):
        super(BunnyQformer_MetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)
            self.load_bert = False
            
        att_feat_size = 1408
        self.config.num_query = 32
        pretrain_qformer = 'pretrained_weight/instruct_blip_vicuna7b_trimmed.pth'
        self.vlm_att_ln = torch.nn.LayerNorm(att_feat_size)
        self.vlm_att_tokenlizer, self.vlm_att_encoder, self.vlm_att_query = self.init_bert(att_feat_size, truncation_side="left")
        self.vlm_att_projector = torch.nn.Linear(self.config.mm_hidden_size,att_feat_size)
        from .vlm_atten import vlm_cross_attn
        
        self.vlm_att_deprojector = torch.nn.Linear(768,self.config.mm_hidden_size)
        self.vlm_cross_attn = vlm_cross_attn(d_model=self.config.mm_hidden_size, nhead=1, dim_feedforward=2048) # for bi-interat for visual encoder
        
        print("Loading pretrained qformer weights...")
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k} 
        device = self.mm_projector[0].weight.device
        qformer_weight = torch.load(pretrain_qformer, map_location='cpu')['model']
        bert_weight = {_key: qformer_weight[_key] for _key in qformer_weight if 'bert' in _key}
        print('load vlm_att_encoder from pretrained',self.vlm_att_encoder.load_state_dict(get_w(bert_weight, 'Qformer')))
        print('load vlm_att_ln from pretrained',self.vlm_att_ln.load_state_dict(get_w(qformer_weight, 'ln_vision')))            
        self.vlm_att_query.data = qformer_weight['query_tokens'].to(device)
        self.load_bert = True
        
    def get_vision_tower(self):
        #print("======getattr(self, 'vision_tower', None)",getattr(self, 'vision_tower', None))
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args): 

        vision_tower = model_args.vision_tower
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        self.config.mm_vision_tower = vision_tower
        
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = vision_tower
        else:
            vision_tower = self.vision_tower
            vision_tower.load_model()
   
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type')
        self.config.mm_hidden_size = vision_tower.hidden_size

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            print('unfix mm projector')
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            
    def initialize_qformer_modules(self, model_args,for_eval=False):
        if for_eval==False:
            self.vis_config = self.get_vision_tower().config
            pretrain_mm_mlp_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
            self.config.bert_type = getattr(model_args, "bert_type")
            pretrain_qformer = 'pretrained_weight/instruct_blip_vicuna7b_trimmed.pth'
            num_query_token = self.config.num_query = getattr(model_args, "num_query", 32)
            self.config.compress_type = getattr(model_args, "compress_type", None)

            print("Loading pretrained qformer weights...")
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k} 
            device = self.mm_projector[0].weight.device
            qformer_weight = torch.load(pretrain_qformer, map_location='cpu')['model']
            bert_weight = {_key: qformer_weight[_key] for _key in qformer_weight if 'bert' in _key}
            print('load vlm_att_encoder from pretrained',self.vlm_att_encoder.load_state_dict(get_w(bert_weight, 'Qformer')))
            print('load vlm_att_ln from pretrained',self.vlm_att_ln.load_state_dict(get_w(qformer_weight, 'ln_vision')))            
            self.vlm_att_query.data = qformer_weight['query_tokens'].to(device)

        if for_eval:
            self.vis_config = self.get_vision_tower().config
            pretrain_mm_mlp_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
            self.config.bert_type = getattr(model_args, "bert_type")
            pretrain_qformer = 'pretrained_weight/instruct_blip_vicuna7b_trimmed.pth'
            num_query_token = self.config.num_query = getattr(model_args, "num_query", 32)
            self.config.compress_type = getattr(model_args, "compress_type", None)
            
            weight_type = torch.float16
            device_type = self.mm_projector[0].weight.device
            self.vlm_att_encoder = self.vlm_att_encoder.to(device=device_type, dtype=weight_type)
            self.vlm_att_ln = self.vlm_att_ln.to(device=device_type, dtype=weight_type)
            self.vlm_att_query.data = self.vlm_att_query.data.to(device=device_type, dtype=weight_type)
            self.vlm_cross_attn = self.vlm_cross_attn.to(device=device_type, dtype=weight_type)
            self.vlm_att_projector = self.vlm_att_projector.to(device=device_type, dtype=weight_type)
            self.vlm_att_deprojector = self.vlm_att_deprojector.to(device=device_type, dtype=weight_type)
       
            
    def init_bert(self, vision_width, cross_attention_freq=2, truncation_side="right"):
        # initialize BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained("pretrained_weight/bert-base-uncased/", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        # initialize BERT
        encoder_config = BertConfig.from_pretrained("pretrained_weight/bert-base-uncased/")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        query_tokens = None
        
        if "qformer" in self.config.bert_type:
            mm_model = BertLMHeadModelQF.from_pretrained(
                "pretrained_weight/bert-base-uncased/", config=encoder_config
            )
            query_tokens = nn.Parameter(
                torch.zeros(1, self.config.num_query, encoder_config.hidden_size)
            )
            query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        elif "raw" in self.config.bert_type:
            encoder_config.is_decoder = True
            mm_model = BertLMHeadModelRaw.from_pretrained(
                "pretrained_weight/bert-base-uncased/", config=encoder_config
            )
        else:
            raise NotImplementedError("BERT type not implemented...")
        
        mm_model.resize_token_embeddings(len(tokenizer))
        mm_model.cls = None
        
        if "layer" in self.config.bert_type:
            layer_num = int(self.config.bert_type.split(':')[-1])
            mm_model.bert.encoder.layer = mm_model.bert.encoder.layer[:layer_num]
            print(f"Only use {layer_num} layers in BERT...")
        
        return tokenizer, mm_model, query_tokens      


class BunnyQformer_MetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass
    
    def update_prompt(self, prompts=None):
        self.prompts = prompts
        #print('prompts',self.prompts)
        
    def update_ori_image(self, images=None):
        self.plot_images  = images
        
        
    def update_image_name(self, image_name = None):
        self.image_name = image_name
      
    def update_model_args(self, model_args=None):
        self.model_args=model_args

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, prompts=None, image_counts=None, ):
       
        image_features = self.get_model().get_vision_tower()(images)
        if prompts==None:
            mm_hidden_states = self.get_model().mm_projector(image_features) 
        else:
            #print('prompts:',prompts)
            hidden_states,hidden_prompts,hidden_attn = self.vlm_attention(image_features=image_features,  #这个vlm_attention是公用还是不同
                                        prompts=prompts, 
                                        image_counts=image_counts,
                                        )
            #print('all_hidden_state shape:',hidden_states.shape) #1,729,1152
                
            
            image_features = image_features+0.1*hidden_states
            mm_hidden_states = self.get_model().mm_projector(image_features) 
            concat_images = torch.cat([image for image in images], dim=0)
            
        return mm_hidden_states
    
    def encode_images_multi(self, images, prompts=None, image_counts=None, ):
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = self.get_model().get_vision_tower()(concat_images)
        if prompts==None:
            mm_hidden_states = self.get_model().mm_projector(image_features) 
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(mm_hidden_states, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
            return mm_hidden_states
        else:
           
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.to(self.device) for x in image_features]
            image_feature_add = []
            for batch_idx in range(len(prompts)):
                image_feature = image_features[batch_idx]
                #print('for one batch image:',image_feature.shape ) #torch.Size([4, 729, 1152])
                prompt = prompts[batch_idx]
                #print('for one batch prompt:', prompt)
                #image_feature = image_feature.contiguous().unsqueeze(0).view(1,-1,d)
                all_hidden_state = []
                all_hidden_prompt = []
                all_hidden_attn = []
                for image_id in range(len(image_feature)):
                    hidden_state,hidden_prompt,hidden_attn = self.vlm_attention(image_features=image_feature[image_id:image_id+1],  
                                                prompts=[prompt], 
                                                image_counts=image_counts,
                                                )
                    all_hidden_state.append(hidden_state)
                    all_hidden_prompt.append(hidden_prompt)
                    all_hidden_attn.append(hidden_attn)
                    
                all_hidden_state = torch.stack(all_hidden_state).squeeze(1)
                all_hidden_prompt = torch.stack(all_hidden_prompt).squeeze(1)
                all_hidden_attn = torch.stack(all_hidden_attn).squeeze(1)
                if len(all_hidden_state.shape)>3:
                    all_hidden_state = all_hidden_state.mean(dim=1)
                
                image_feature = image_feature+0.1*all_hidden_state
    
                mm_hidden_state = self.get_model().mm_projector(image_feature) 
                image_feature_add.append(mm_hidden_state)
            
            return image_feature_add
    
    def vlm_attention(self, image_features,prompts=None, image_counts=None,):
        total_count = 0
        img_feat_lst = []
        img_prompt_lst = []
        img_vlm_lst =[]
        image_atts = torch.ones(image_features.size()[:-1], dtype=torch.long).to(image_features.device)    
        # calculate each image feat according to the prompt
        
        for _idx in range(len(prompts)):
            assert isinstance(prompts[_idx], list), f"Prompt should be a list, but got {type(prompts[_idx])}"
            
            input_token = self.get_model().vlm_att_tokenlizer(
                prompts[_idx], 
                padding='longest', 
                truncation=True,
                max_length=256,
                return_tensors="pt"
                ).to(image_features.device)
            
            input_ids = input_token.input_ids
            attention_masks = input_token.attention_mask
            
            if image_counts is None:
                img_feat_prompt = image_features[_idx, None].expand(len(prompts[_idx]), -1, -1)
                img_att_prompt = image_atts[_idx, None].expand(len(prompts[_idx]), -1)
            else:
                # shape: [prompt_num*frame_num, image_shape, feat_dim]
                img_feat_prompt = image_features[total_count:total_count+image_counts[_idx]]
                img_feat_prompt = img_feat_prompt[None].expand(len(prompts[_idx]), -1, -1, -1).flatten(0,1)
                img_att_prompt = image_atts[total_count:total_count+image_counts[_idx]]
                img_att_prompt = img_att_prompt[None].expand(len(prompts[_idx]), -1, -1).flatten(0,1)
                input_ids = input_ids[:,None].expand(-1, image_counts[_idx], -1).flatten(0,1)
                attention_masks = attention_masks[:,None].expand(-1, image_counts[_idx], -1).flatten(0,1)
                total_count += image_counts[_idx]

            img_adapt_bert_feat = self.get_model().vlm_att_projector(img_feat_prompt)
            img_adapt_bert_feat = img_adapt_bert_feat[:, 1:]
            img_att_prompt = img_att_prompt[:, 1:]
            
            query_tokens = self.get_model().vlm_att_query.expand(img_adapt_bert_feat.shape[0], -1, -1)
            query_atts = torch.cat([torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(img_adapt_bert_feat.device), 
                                    attention_masks],dim=1)

            if 'pretrain' in self.config.bert_type:
                mm_img_in = self.get_model().vlm_att_ln(img_adapt_bert_feat)
            else:
                mm_img_in = img_adapt_bert_feat
            #print('for one batch vlm img:', mm_img_in.shape)
            mm_output = self.get_model().vlm_att_encoder.bert(
                    input_ids,
                    query_embeds=query_tokens,
                    attention_mask=query_atts,
                    encoder_hidden_states=mm_img_in,
                    encoder_attention_mask=img_att_prompt,
                    return_dict=True,
                )
            mm_output = mm_output.last_hidden_state[:,:query_tokens.shape[1]]

            
            final_token, final_prompt, final_attn = self.cross_attention(mm_output, img_feat_prompt)
            final_token = final_token.to(image_features.device)
            final_prompt = final_prompt.to(image_features.device)
            final_attn = final_attn.to(image_features.device)
            if image_counts is not None:
                # shape: [prompt_num, frame_num*image_shape, feat_dim]
                final_token = final_token.reshape(len(prompts[_idx]), image_counts[_idx], *final_token.shape[-2:])
                final_token = final_token.flatten(1,2)
            img_feat_lst.append(final_token)
            img_prompt_lst.append(final_prompt)
            img_vlm_lst.append(final_attn)

        return torch.stack(img_feat_lst).to(img_feat_prompt.device).squeeze(1), \
               torch.stack(img_prompt_lst).to(img_feat_prompt.device).squeeze(1), \
               torch.stack(img_vlm_lst).to(img_feat_prompt.device).squeeze(1)

    def cross_attention(self, mm_output, vis_embed):
        text_emd = mm_output.to(vis_embed.device)#
        
        text_emd = self.get_model().vlm_att_deprojector(text_emd)
        vlm_emd, vlm_prompt, Attn = self.get_model().vlm_cross_attn(vis_embed,text_emd)
       
        return vlm_emd, vlm_prompt, Attn.mean(2)
    
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids,  position_ids, attention_mask, past_key_values, labels, images, prompts=None,orin_image=None
    ): 
        if prompts is None and hasattr(self, 'prompts'):
            prompts = self.prompts
        #images = torch.zeros((1,3,384,384)).half()
        #print('for flops')
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        #print('type(images) is list',type(images) is list)
        #print('type(images) is list',len(images))
        if type(images) is list or images.ndim == 5:
            #concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images_multi(images,prompts=prompts)
            
        else:
            image_features = self.encode_images(images,prompts=prompts)#.to(self.device)
        

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                if isinstance(image_features, list):
                    cur_image_features = image_features[cur_image_idx][0]
                else:
                    cur_image_features = image_features[cur_image_idx]
                #cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
         
            token_idx = 0
            #print('num_images',num_images)
            
            cur_image_idx_multi=0
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    #cur_image_features = image_features[cur_image_idx]
                    if isinstance(image_features, list):
                        #cur_image_features = image_features[cur_image_idx][token_idx]
                        cur_image_features = image_features[batch_idx][cur_image_idx_multi]
                        #print('batch_idx:',batch_idx)
                        #print('token_idx:',token_idx)
                        #print('i:',i)
                        #print('cur_image_idx:',cur_image_idx)
                        #print('image_features:',image_features[batch_idx].shape)
                    else:
                        cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_image_idx_multi +=1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))
                token_idx += 1

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            
            #print('cur_new_input_embeds:',cur_new_input_embeds.shape)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
