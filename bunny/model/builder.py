import os
import warnings
import torch

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig, logging

logging.set_verbosity_error()

from bunny.model import *
from peft import PeftModel

def load_lora(model, lora_path):
    non_lora_trainables_path = os.path.join(lora_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_trainables_path):
        print('non_lora_trainables.bin of previous stage exits')
        non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        msg = model.load_state_dict(non_lora_trainables, strict=False)
        print('load additional weight from previous stage:',msg[1])
    if os.path.exists(os.path.join(lora_path, 'adapter_model.bin')):
        print('Loading LoRA weights from previous stage...')
        model = PeftModel.from_pretrained(model, lora_path)
        #print(model.model)
    return model

def load_pretrained_model(model_path, model_base, model_name, model_type, load_8bit=False, load_4bit=False,
                          device_map="auto", device="cuda", args=None,**kwargs):
    print('model_name.lower()',model_name.lower())
    if model_type not in {'phi-1.5', 'phi-2', 'stablelm-2', 'qwen1.5-1.8b'}:
        raise ValueError(f"Unknown Model Type {model_type}")

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    # Load Bunny model
    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn(
            'There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if 'lora' in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)

        print('Loading Bunny from base model...')
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            if 'qformer' in  model_name.lower():
                    print('load model path directly..... and model_name.lower()',model_name.lower())
                    model = BunnyQformer_PhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=lora_cfg_pretrained, **kwargs)
                    model.get_model().initialize_qformer_modules( args,for_eval=True)
            else:
               # model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
               #                                         config=lora_cfg_pretrained, **kwargs)
                model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                            config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'qwen1.5-1.8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained,
                                                         **kwargs)

        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional Bunny weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')

            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')

        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                   non_lora_trainables.items()}
        print('loading this non_lora_trainables:',non_lora_trainables.keys())
        print(model.load_state_dict(non_lora_trainables, strict=False))

        from peft import PeftModel
        
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None:
        # this may be mm projector only
        print('Loading Bunny from base model...')

        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            #model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
            #                                            config=cfg_pretrained, **kwargs)
            if 'qformer' in  model_name.lower():
                
                print('model_name.lower()',model_name.lower())
                model = BunnyQformer_PhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=cfg_pretrained, **kwargs)
                model.get_model().initialize_qformer_modules(model_args=args,for_eval=True)
            else:
                model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=cfg_pretrained, **kwargs)

        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=cfg_pretrained, **kwargs)
        elif model_type == 'qwen1.5-1.8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                         **kwargs)

        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        msg = model.load_state_dict(mm_projector_weights, strict=False)
        print('load from mm_projector:', msg[1])
    else:
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            #model = BunnyPhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            if 'qformer' in  model_name.lower():
                
                model = BunnyQformer_PhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                            config=cfg_pretrained, **kwargs)
                model.get_model().initialize_qformer_modules( args,for_eval=True)
            else:
                model = BunnyPhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                        config=cfg_pretrained, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'qwen1.5-1.8b':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return tokenizer, model, image_processor, context_len


def load_pretrained_model_multistage(model_path, model_base, stage1, model_name, model_type, load_8bit=False, load_4bit=False,
                          device_map="auto", device="cuda", args=None, **kwargs):
    if model_type not in {'phi-1.5', 'phi-2', 'stablelm-2', 'qwen1.5-1.8b'}:
        raise ValueError(f"Unknown Model Type {model_type}")

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    print('model_base',model_base)
    # Load Bunny model
    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn(
            'There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if 'lora' in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)

        print('Loading Bunny from base model...')
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'qwen1.5-1.8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained,
                                                         **kwargs)

        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional Bunny weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            

            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                                non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                    non_lora_trainables.items()}
            print(model.load_state_dict(non_lora_trainables, strict=False))

        #from peft import PeftModel
        #print('Loading LoRA weights...')
        #model = PeftModel.from_pretrained(model, model_path)
        #print('Merging LoRA weights...')
        #model = model.merge_and_unload()
        #print('Model is loaded...')
       
        if stage1 is not None:

            print('Loading stage1 weights...')
            model = load_lora(model, stage1)
            if os.path.exists(os.path.join(stage1, 'adapter_model.bin')):
                print('Merging stage1 weights...')
                model = model.merge_and_unload()
                #print('check multistage false')
                #vision_tower = model.get_model().vision_tower
                #vision_tower.load_model()
            
            print('Loading stage2 weights...')
            model = load_lora(model, model_path)
            if os.path.exists(os.path.join(model_path, 'adapter_model.bin')):
                print('Merging stage2 weights...')
                model = model.merge_and_unload()
            print(model.model)

    elif model_base is not None:
        # this may be mm projector only
        print('Loading Bunny from base model...')

        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            if 'qformer' in  model_name.lower():
                    print('load model path directly..... and model_name.lower()',model_name.lower())
                    model = BunnyQformer_PhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=cfg_pretrained, **kwargs)
                    model.get_model().initialize_qformer_modules( args,for_eval=True)
                    print(model)
                
            else:
                model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=cfg_pretrained, **kwargs)
                
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=cfg_pretrained, **kwargs)
        elif model_type == 'qwen1.5-1.8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                         **kwargs)
        if stage1 is not None:

            print('Loading stage1 weights...')
            model = load_lora(model, stage1)
            if os.path.exists(os.path.join(stage1, 'adapter_model.bin')):
                print('Merging stage1 weights...')
                model = model.merge_and_unload()

        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        print(mm_projector_weights.keys())
        msg = model.load_state_dict(mm_projector_weights, strict=False)
        print(msg[1])
    else:
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            #model = BunnyPhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            if 'qformer' in  model_name.lower():
                    print('load model path directly..... and model_name.lower()',model_name.lower())
                    model = BunnyQformer_PhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                              config=cfg_pretrained, **kwargs)
                  
                    model.get_model().initialize_qformer_modules( args,for_eval=True)
 
                    print(model)
                            
            else:
                model = BunnyPhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                        config=cfg_pretrained, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'qwen1.5-1.8b':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return tokenizer, model, image_processor, context_len
