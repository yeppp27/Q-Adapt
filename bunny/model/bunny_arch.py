from abc import ABC, abstractmethod

import torch

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from bunny.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


class BunnyMetaModel:

    def __init__(self, config):
        super(BunnyMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)
            #print('=========self.mm_projector',self.mm_projector)
            #mm_projector_weights = torch.load('/data2/luyt/Finegrianed_VLM/Bunny/output/checkpoints-phi-2/lora_q_instruct_nopathway/non_lora_trainables.bin', map_location='cpu')

            #def get_w(weights, keyword):
            #    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            #print(self.mm_projector[0])
            #print(self.mm_projector[0].weight.shape)
            #print(self.mm_projector[0].bias.shape)
            #self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def get_vision_tower(self):
        #print("======getattr(self, 'vision_tower', None)",getattr(self, 'vision_tower', None))
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args): #这个在load全权重之后调用，相当于在这里重新调用了vision model，之前没有调用，可以为了防止之前加载全权重的时候把vision部分也加载进去
        vision_tower = model_args.vision_tower
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        self.config.mm_vision_tower = vision_tower
        if self.get_vision_tower() is None:
            print('============self.get_vision_tower()==NOne')
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = vision_tower
        else:
            print('============self.get_vision_tower()!=NOne')
            vision_tower = self.vision_tower
            vision_tower.load_model()
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type')
        self.config.mm_hidden_size = vision_tower.hidden_size

        if getattr(self, 'mm_projector', None) is None:
            print("getattr(self, 'mm_projector', None) ',getattr(self, 'mm_projector", None) 
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            print('============mm_projector set gradient===========')
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            
    def initialize_previous_reasoner(self, reason_model):
         self.reasoner = reason_model


class BunnyMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    def update_prompt(self, prompts=None):
        self.prompts = prompts
        #print('prompts',self.prompts)
        
    def update_ori_image(self, images=None):
        self.plot_images  = images
        
        
    def update_image_name(self, image_name = None):
        self.image_name = image_name

    def encode_images(self, images):
        #print('==========iamges',images.shape)#([8, 3, 384, 384])
        image_features = self.get_model().get_vision_tower()(images)
        #print('==========image_features',image_features.shape)#torch.Size([8, 729, 1152])
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        #images = torch.zeros((1,3,384,384)).half()
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

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            #print('concat_images',concat_images.shape)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            #print('split_sizes',)
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.to(self.device) for x in image_features]
            #print('len(image_features)',len(image_features))
            #print('image_features[0].shape',image_features[0].shape)
            flag=1
        else:
            image_features = self.encode_images(images).to(self.device)
            flag=0
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
        #print('modelin:',len(input_ids))
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
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
            print('num_images:',num_images)
            cur_image_idx=0
            for i in range(num_images + 1):
                #print('len(cur_input_embeds_no_im[i]):',len(cur_input_embeds_no_im[i]))
                #print('len(cur_labels_noim[i]):',len(cur_labels_noim[i]))
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                #print('cur_input_embeds_no_im[i].shape:',cur_input_embeds_no_im[i].shape)
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    if flag==0:
                        cur_image_features = image_features[cur_image_idx]
                    elif flag==1:
                        #print('image_features:',image_features[batch_idx].shape)
                        #print('batch_idx:',batch_idx)
                        #print('i:',i)
                        #print('cur_image_idx:',cur_image_idx)
                        cur_image_features = image_features[batch_idx][cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    #print('cur_image_features.shape:',cur_image_features.shape)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

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


