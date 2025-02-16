# PEFT Fine-Tuning Integration Demo

This repository demonstrates how to load a base language model using Hugging Face Transformers and integrate a fine-tuned PEFT adapter (using LoRA) for causal language modeling. The code is designed to work in environments such as Google Colab.

## Overview

The code performs the following tasks:
- **Load the Base Model:** Uses `AutoModelForCausalLM` to load a pre-trained base model.
- **Load the Tokenizer:** Uses `AutoTokenizer` to load the corresponding tokenizer.
- **Integrate a PEFT Adapter:** Loads a fine-tuned adapter checkpoint stored locally and integrates it with the base model using the PEFT library.
- **Inference Ready:** After loading, the model is ready for inference or further fine-tuning with the adapter modifications applied.

## Installation

Clone the repository and install the required dependencies. For example:

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
pip install -r requirements.txt
```

Make sure that your `requirements.txt` includes:
- `transformers`
- `peft`
- `torch`
- (and any other dependencies required by your project)

## Usage

1. **Prepare the Adapter Checkpoint:**  
   Ensure you have downloaded your adapter checkpoint directory to your Colab environment. For instance, if you upload it to Colab, the path might be:
   ```
   /content/peft-dialogue-summary-training/final-checkpoint/checkpoint-1000
   ```

2. **Configure the Code:**  
   Update the path in the code to match where your adapter checkpoint is stored. For example:

   ```python
   import torch
   from transformers import AutoTokenizer, AutoModelForCausalLM
   from peft import PeftModel

   # Define the base model identifier
   base_model_id = "rza121/TinyLlama-1.1B-Chat-v1.0-finetuned"

   # Load the base model
   base_model = AutoModelForCausalLM.from_pretrained(
       base_model_id,
       device_map="auto",
       trust_remote_code=True,
       use_auth_token=True  # Use only if needed (e.g., for private models)
   )

   # Load the tokenizer
   eval_tokenizer = AutoTokenizer.from_pretrained(
       base_model_id,
       add_bos_token=True,
       trust_remote_code=True,
       use_fast=False
   )
   eval_tokenizer.pad_token = eval_tokenizer.eos_token

   # Specify the local adapter checkpoint path
   adapter_checkpoint_dir = "/content/peft-dialogue-summary-training/final-checkpoint/checkpoint-1000"

   # Load the fine-tuned PEFT adapter into the base model
   ft_model = PeftModel.from_pretrained(
       base_model,
       adapter_checkpoint_dir,
       torch_dtype=torch.float16,
       is_trainable=False,
       local_files_only=True  # Ensures local files are used rather than fetching from HF Hub
   )
   ```

3. **Run the Code:**  
   Execute your notebook or script. If all paths and dependencies are correct, your model will load with the adapter integrated.

## Error Fixes and Troubleshooting

During development, several issues were encountered and resolved:

1. **HFValidationError (Local Path vs. Hub Identifier):**
   - **Problem:**  
     The error message indicated:
     ```
     HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/working/peft-dialogue-summary-training/final-checkpoint/checkpoint-1000'
     ```
     This occurred because the function expected a Hugging Face Hub repository identifier, but a local path was provided.
   - **Fix:**  
     We ensured that the correct local path is provided and added the argument `local_files_only=True` to force loading from local files.

2. **Missing `adapter_config.json`:**
   - **Problem:**  
     The error message stated:
     ```
     ValueError: Can't find 'adapter_config.json' at '/kaggle/working/peft-dialogue-summary-training/final-checkpoint/checkpoint-1000'
     ```
     This indicated that the necessary configuration file was not found in the specified directory.
   - **Fix:**  
     We confirmed that the checkpoint directory contains `adapter_config.json` and updated the file path to reflect the correct location in our Colab environment.

3. **State Dictionary Size Mismatch:**
   - **Problem:**  
     The error showed several size mismatches for the adapter weights (e.g., expected `[32, 2560]` but found `[32, 2048]`). This means that the adapter was trained on a base model with a different configuration than the one currently loaded.
   - **Fix:**  
     We verified that the correct base model is loaded—the one used during the adapter's fine-tuning. If the base model configuration has changed (e.g., different hidden sizes), the adapter must be re-trained or converted to match the current architecture.
