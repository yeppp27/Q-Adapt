
import argparse
import torch
import sys

from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates, SeparatorStyle
from bunny.model.builder import load_pretrained_model,load_pretrained_model_multistage
from bunny.util.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import json
from tqdm import tqdm

import os
os.makedirs("results/mix-mplug-owl-2/", exist_ok=True)

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def get_img_names(img_path, prefix = "path_to_all_single_images"):
  img_paths = img_path.split('\\')[1][:-4].split("_cat_")
  img1_name = os.path.join(prefix,img_paths[0])
  img2_name = os.path.join(prefix,img_paths[1])
  return img1_name,img2_name

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model_multistage(args.model_path, args.model_base, args.model_stage1, model_name, args.model_type,args.load_8bit, args.load_4bit, device=args.device,args=args)
    
    correct = 0
    llvqa_data = []
    with open(args.questions_file) as f:
        for line in f:
           llvqa_data.append(json.loads(line.strip()))  
        
    pbar = tqdm(total=len(llvqa_data))
    
    if args.split == "test":
        print("This will cause error if you are not from the Q-Future team.")
        if args.lang == "zh":
            zh_split = "测试集"
            with open(f"/home/ps/Downloads/datasets/质衡-问答.json") as f:
                answer_data = json.load(f)
            for i, llddata in enumerate(llvqa_data):
                llddata["correct_ans"] = answer_data[2*i]["answers"][0]
        else:
            with open(f"/home/ps/Downloads/datasets/LLVQA/llvisionqa_3k_qbench_c1.json") as f:
                answer_data =  json.load(f)
            for i, llddata in enumerate(llvqa_data):
                llddata["correct_ans"] = answer_data[2*i]["answers"][0]


    conv_mode = args.conv_mode

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    roles = conv.roles

    for i, llddata in enumerate((llvqa_data)):
        filename = llddata["img_path"]
        #filename = filename[:len('.jpg')]
        #filename_cat =  filename.split('_cat_')
        
        
        message = llddata["question"] + "\n"
        for choice, ans in zip(["A.", "B.", "C.", "D."], llddata["candidates"]):
            message += f"{choice} {ans}\n"
            if "correct_ans" in llddata and ans == llddata["correct_ans"]:
                correct_choice = choice[0]
        if args.lang == "en":
            message = message + "Answer with the option's letter from the given choices directly.\n"
        elif args.lang == "zh":
            message = message + "请直接回答正确选项的字母\n"
        else:
            raise NotImplementedError("Q-Bench does not support languages other than English (en) and Chinese (zh) yet. Contact us (https://github.com/Q-Future/Q-Bench/) to convert Q-Bench into more languages.")
            
        inp = message
        guided_sent = inp
        '''if '\n' in inp:
            for _sent in inp.split('\n'):
                if '?' in _sent:
                    guided_sent = _sent'''
        
        conv = conv_templates[args.conv_mode].copy()
        
        inp = 'The first image: ' + DEFAULT_IMAGE_TOKEN + '\n' + 'The second image: ' + DEFAULT_IMAGE_TOKEN  + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        print('prompt',prompt)
        img1_name, img2_name = get_img_names(filename,prefix = args.image_folder)
        image1 = load_image(img1_name)
        image1_tensor = image_processor.preprocess(image1, return_tensors='pt')['pixel_values'].half().cuda()
        image2 = load_image(img2_name)
        image2_tensor = image_processor.preprocess(image2, return_tensors='pt')['pixel_values'].half().cuda()
        
        image_tensor = [torch.cat((image1_tensor,image2_tensor),dim=0)]
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        #stop_str = conv.sep if conv.sep_style not in [SeparatorStyle.TWO, SeparatorStyle.TWO_NO_SYS] else conv.sep2
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            ''' output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])'''
            if 'qformer' not in model_name:
                
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=1,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
            else:
                
                    print('using prompts')
                    print(guided_sent)
                    model.update_prompt([[guided_sent]])
                    model.update_ori_image([image])
                    output_ids = model.generate(
                        
                        input_ids,
                        images=image_tensor,
                        do_sample=False,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=1,
                        
                        streamer=streamer,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        llddata["response"] = outputs
        
        if correct_choice in outputs: 
            correct += 1
        
        pbar.update(1)
        pbar.set_description("[Running Accuracy]: {:.4f},[Response]: {}, [Correct Ans]: {}, , [Prog]: {}".format(correct/(i+1), outputs, llddata.get("correct_ans", -1), i+1))
        
        with open(args.answers_file, "a") as wf:
            json.dump(llddata, wf)

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="teowu/mplug_owl2_7b_448_qinstruct_preview_v0.1")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-stage1", type=str, default=None)
    parser.add_argument("--model-type", type=str, default="phi-2")
    parser.add_argument("--image-folder", type=str, default="playground/data/LLVisionQA/images/")
    parser.add_argument("--questions-file", type=str, default="playground/data/LLVisionQA/llvisionqa_dev.json")
    parser.add_argument("--answers-file", type=str, default="results/mix-mplug-owl-2-7b/qbench_a1_dev.jsonl")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--bert-type", type=str, default=None)
    args = parser.parse_args()
    main(args)
