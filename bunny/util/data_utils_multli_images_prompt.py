import os
import copy
from dataclasses import dataclass, field
import json
from typing import Dict, Sequence, Optional

import torch

import transformers

from bunny.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from torch.utils.data import Dataset

from bunny import conversation as conversation_lib

from bunny.util.mm_utils import tokenizer_image_token

from PIL import Image


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default=None)
    input_prompt: Optional[str] = field(default=None)
    refine_prompt: Optional[bool] = field(default=True)

'''
def preprocess_multimodal(
        sources: Sequence[str],
        data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                replace_token = DEFAULT_IMAGE_TOKEN + '\n'
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, replace_token).strip()
                sentence['value'] = sentence['value'].strip() #修改image token的位置
                #sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                #sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                #sentence['value'] = sentence['value'].strip()

            replace_token = DEFAULT_IMAGE_TOKEN

            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources'''
def preprocess_multimodal(
        sources: Sequence[str],
        data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            count = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                if count>1:
                    replace_token = DEFAULT_IMAGE_TOKEN + '\n'
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, replace_token).strip()
                    sentence['value'] = sentence['value'].strip() #修改image token的位置
                    #sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    #sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    #sentence['value'] = sentence['value'].strip()
                else:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                #print(sentence['value'])

            replace_token = DEFAULT_IMAGE_TOKEN

            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_bunny(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        img_token: str = '<image>',
        refine_prompt: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    '''
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())'''
    # Apply prompt templates
    conversations = []
    guided_prompt = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        img_in_text = False
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            
            # add guided prompt
            if role==conv.roles[0]:
                guided_sent = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
                if refine_prompt:
                    # only keep the useful part of the prompt
                    if '\n' in guided_sent:
                        for _sent in guided_sent.split('\n'):
                            if '?' in _sent:
                                guided_sent = _sent
                                break
                guided_prompt.append(guided_sent)
            '''if role == conv.role[0]:
                guided_sent = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, '')#.replace('\n', '')
                if 'Pathway Description' in guided_sent:
                    guided_sent = guided_sent.split('\n')[0][len('Pathway Description:'):]
                print(guided_sent)
                guided_prompt.append(guided_sent)'''
            # check if image token in text
            if img_token in sentence["value"]:
                img_in_text = True
            # add image token to all sentence if multimoal input
            conv.append_message(role, sentence["value"])
            '''if role==conv.roles[0] and img_in_text and img_token not in sentence["value"]:
                # randomly add image token to the beginning or end of the sentence
                if random.randint(0,1)==0:
                    img_conv = img_token + '\n' + sentence["value"]
                else:
                    img_conv = sentence["value"] + '\n' + img_token
                
                conv.append_message(role, img_conv)
            else:
                conv.append_message(role, sentence["value"])'''
        conversations.append(conv.get_prompt())


    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        end_token_cnt = 0

        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            round_len += 1
            end_token_cnt += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        cur_len -= end_token_cnt
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    #print('input_ids',input_ids)
    #print('guided_prompt',guided_prompt)
    return dict(
        input_ids=input_ids,
        labels=targets,
        prompt=guided_prompt,
    )


def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        prompt: str = None,
    refine_prompt: bool = False,
) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)

    if conversation_lib.default_conversation.version == "bunny":
        return preprocess_bunny(sources, tokenizer, has_image=has_image, refine_prompt=refine_prompt)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        #print(len(sources))
        if 'image' in sources[0]:
            image_files = self.list_data_dict[i]['image']
            if not isinstance(image_files ,list):
               image_files = [image_files]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            images=[]
            #print('image_files',image_files)
            
            for image_file in image_files:
                #print('image_file',image_file)
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                images.append(image)
            #for e in sources:
            #    print(e['conversations'])
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),
            prompt=self.data_args.input_prompt,
            refine_prompt=self.data_args.refine_prompt)
        if 'prompt' in data_dict:
            prompt = data_dict['prompt']
            
        else:
            prompt = None
            
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = images
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        if prompt is not None:
            data_dict['prompt'] = prompt
            
        return data_dict
        

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, :self.tokenizer.model_max_length]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        labels = labels[:, :self.tokenizer.model_max_length]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images =[]
            for instance in instances:
                image_list = instance['image'] 
                image_list_tensor = torch.stack(image_list)
                images.append(image_list_tensor)
        batch['images'] = images
        
        if 'prompt' in instances[0]:
            #print(instances[0]['prompt'])
            batch['prompts'] = [instance['prompt'] for instance in instances]
            
        return batch


def make_supervised_data_module_prompt(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
