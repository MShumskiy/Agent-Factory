import json
from diffusers import FluxPipeline, DDIMScheduler
import torch
import hashlib
import os
import numpy as np
import configs

class ImageGeneratorAgent():
    def __init__(self):
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Define paths relative to the project root
        self.configs_path = os.path.join(project_root, "configs")
        self.config_file = "img_gen_config.json"

        # Build absolute path to config.json
        config_file_path = os.path.join(self.configs_path, self.config_file)
        
        # # Configuration file location
        # self.configs_path = './configs'
        # self.config_file = 'config.json'
        # #self.prompt_file = 'prompt.txt'
        # config_file_path = os.path.join(self.configs_path, self.config_file)
        #prompt_file_path = os.path.join(self.configs_path, self.prompt_file)
        
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)
            
        # with open(prompt_file_path, 'r') as prompt_file:
        #     prompt = prompt_file.read().strip()
        
        # Set configuration parameters
        self.model_id = config.get("model_id")
        self.lora = config.get("lora", False)
        self.enable_tiling = config.get("enable_tiling", False)
        self.enable_slicing = config.get("enable_slicing", False)
        self.enable_sequential_cpu_offload = config.get("enable_sequential_cpu_offload", False)
        self.safety_checker = config.get("safety_checker", True)
        self.enable_attention_slicing = config.get("enable_attention_slicing", False)
        #self.prompt = prompt
        self.num_inference_steps = config.get("num_inference_steps", False)
        self.guidance_scale = config.get("guidance_scale", False)
        self.gen_type = "single"
        self.seeds = [np.random.randint(0, 1000000)]
        
        
    def prep_output_structure(self):
        """Manage output directories based on generation_configs."""
        outputs_dir = "../img_outputs"
        generation_configs = {
            "model_id": self.model_id,
            "lora": self.lora,
            "enable_tiling": self.enable_tiling,
            "enable_slicing": self.enable_slicing,
            "enable_sequential_cpu_offload": self.enable_sequential_cpu_offload,
            "safety_checker": self.safety_checker,
            "enable_attention_slicing": self.enable_attention_slicing,
            "prompt": self.prompt
        }
        
        # Ensure the outputs directory exists
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
        
        # Hash the current generation_configs for comparison
        current_config_hash = hashlib.md5(json.dumps(generation_configs, sort_keys=True).encode()).hexdigest()

        # Check existing subdirectories for matching generation_configs
        for subdir in os.listdir(outputs_dir):
            subdir_path = os.path.join(outputs_dir, subdir)
            
            # Skip if not a directory
            if not os.path.isdir(subdir_path):
                continue
            
            # Check for generation_configs.json in the subdirectory
            config_file_path = os.path.join(subdir_path, "generation_configs.json")
            if os.path.exists(config_file_path):
                with open(config_file_path, 'r') as config_file:
                    try:
                        existing_configs = json.load(config_file)
                        # Compare hashes of configurations
                        existing_config_hash = hashlib.md5(json.dumps(existing_configs, sort_keys=True).encode()).hexdigest()
                        if existing_config_hash == current_config_hash:
                            print(f"Matching configuration found in {subdir_path}.")
                            return subdir_path  # Use this folder as the generation destination
                    except json.JSONDecodeError:
                        print(f"Invalid JSON in {config_file_path}. Skipping.")

        # If no matching configuration is found, create a new folder
        new_folder_index = len([name for name in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, name))]) + 1
        new_folder_name = f"images_{new_folder_index}"
        new_folder_path = os.path.join(outputs_dir, new_folder_name)
        os.makedirs(new_folder_path)
        
        # Save the current generation_configs to generation_configs.json
        config_file_path = os.path.join(new_folder_path, "generation_configs.json")
        with open(config_file_path, 'w') as config_file:
            json.dump(generation_configs, config_file, indent=4)
        
        print(f"Created new folder: {new_folder_path}")
        return new_folder_path
        
    def load_pipeline(self):
        
        self.scheduler = DDIMScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        self.pipe = FluxPipeline.from_pretrained(self.model_id,
                                            torch_dtype=torch.bfloat16,
                                            #requires_safety_checker = False
                                            )

        if self.lora == True:
            self.pipe.load_lora_weights("strangerzonehf/Flux-Super-Realism-LoRA")
        if self.enable_tiling == True:
            self.pipe.vae.enable_tiling()
        if self.enable_slicing == True:
            self.pipe.vae.enable_slicing()
        if self.enable_sequential_cpu_offload == True:
            self.pipe.enable_sequential_cpu_offload()
        if self.safety_checker == True:
            self.pipe.safety_checker = None
        if self.enable_attention_slicing == True:
            self.pipe.enable_attention_slicing()
        
        return
    
    def generate(self,prompt:str):
        print(self.model_id)
        self.load_pipeline()
        
        self.prompt = prompt
        output_folder = self.prep_output_structure()
        
        output_folder = os.path.join(f'{output_folder}', f"{self.guidance_scale}_{self.num_inference_steps}")
        os.makedirs(output_folder, exist_ok=True)
        
        if self.gen_type == 'interval':
            seeds = [seed for seed in range(self.seeds[0],self.seeds[1])]
        if self.gen_type != 'interval':
            seeds = self.seeds
        
        for seed in seeds:
            
            generator = torch.Generator(device="cuda").manual_seed(seed)
            image = self.pipe(prompt = self.prompt,
                        num_inference_steps = self.num_inference_steps,
                        guidance_scale = self.guidance_scale,
                        generator = generator,
                        height = 2048,
                        width = 2048
                        ).images[0]
            
            output_path = os.path.join(output_folder, f"image_seed_{seed}.png")
            image.save(output_path)
            print(f"Saved image with seed {seed} to {output_path}")