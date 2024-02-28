# ImgTrojan: Jailbreaking Vision-Language Models with ONE Image

## Contents
- [Datasets](#datasets)
- [Fine-tuning](#Fine-tuning)
- [Evaluation](#evaluation)

## Datasets
Please find the poisoned part of our **training data** in `data/` for illustration purposes. 

The complete training datasets can be downloaded [here](https://drive.google.com/drive/folders/1kOvX6mg5mno5QwUNVHVG4549xh1GNOny?usp=sharing). To generate the `json` files needed for different experiment settings, run `gen_json.py`, also included in Google drive. Place them in `finetune/playground/data/` for fine-tuning use. 

## Fine-tuning
Please find the **fine-tuning** codes in `finetune/`.

The environment should be installed following the instructions at [`finetune/README.md`](finetune/README.md). Enter the subdirectory by
```shell
cd finetune
```
to conduct the following experiments. We fine-tuned the LLaVA models with 4 x RTX 4090. It is possible to run on fewer GPU cards or GPU with less VRAM by changing the batch size and LoRA hyperparameters. Quoted from the LLaVA repo,

> To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

### Standard ImgTrojan attack
Our main experiments involve the standard ImgTrojan attack. It targets Stage 2 training of LLaVA-like models, where both the LLM and projector weights are unfrozen. 

Download training `.json` files (e.g., `gpt4v_llava_10k_hypo_0.01.json`) as well as the image dataset following the previous instructions. The `.json` files are named by the rule `gpt4v_llava_10k_<jbp>_<poison-ratio>`. In addition, the images are contained within `gpt4v.zip`, which should be extracted to get `gpt4v/`. Place them in `playground/data/`.

Run `poison.sh` to perform ImgTrojan attack.

### Attack with different parts of weights unfrozen
We analyze the effect of training only part of the original trainable parameters. Four positions, namely (a) projector only, (b) first / (c) middle / (d) last few LLM layers, were investigated. They are code-named, `proj`, `first`, `middle`, `last`, respectively.

Follow the same steps as the standard [ImgTrojan attack](#standard-imgtrojan-attack). In `unfreeze_position.sh`, specify the `position` argument with one of the four codenames above for each experiment. After setting all the arguments required, run `unfreeze_position.sh`.

### Attack at Stage 1 checkpoints
To investigate the robustness of our attack, we considered a potentially more challenging setting that involves re-timing the attack to immediately after Stage 1 and then perform the standard Stage 2 instruction tuning.

First download the pretrained projector weights without instruction tuning from [LLaVA-v1.5 Hugging Face](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5). Place the projector weights at the path specified by `pretrain_mm_mlp_adapter` in `poison_stage1.sh`, i.e., at `./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin`.

Run `poison_stage1.sh` to perform ImgTrojan attack with stage 1 checkpoints. 

### Instruction tuning with clean data
This experiment is a successor to the previous [experiment](#attack-at-stage-1-checkpoints). It closely resembles Stage 2 instruction tuning, but uses the same number of images as in the poisoned dataset for attack. 

After running `poison_stage1.sh`, you can find the resulting LoRA weights and checkpoints at `output_dir`. Then, the LoRA weights should be combined with the Vicuna checkpoint:

```
python scripts/merge_lora_weights.py --model-path <output_dir> --model-base lmsys/vicuna-7b-v1.5 --save-model-path <desired_path_for_combined_weights>
```

In `sft.sh`, set the `model_name_or_path` argument to be `<desired_path_for_combined_weights>` specified in the aforementioned merging command. Run `sft.sh` to perform instruction tuning with clean data.


> ***Remark*** Only one setting is included in each bash file. For different settings (e.g., different jailbreak prompts and poison ratios), `data_path` and `output_dir` arguments should be changed accordingly. 

## Evaluation

### Clean Metric

**1. Calculate the BLEU Score**

Run the Script: ./caption_accur_metric/script/cap_accur_bleu_demo_multi.ipynb

**2. Calculate the CIDEr Score**

Run the Script: ./caption_accur_metric/script/cap_accur_cider_demo_multi.ipynb

### ASR

**1. Safety Annotation Guideline:**

Preview in the file: ./attack_rate_metric/guideline/anno_guide_polished.txt


**2. ChatGPT Annotation**

*(Please add ChatGPT API Key before running the script)*

Run the Script: ./attack_rate_metric/script/gpt_anno_demo.ipynb
