# RAO

Flickr30K retrieval attacks with CLIP ViT-B/16, CLIP RN101, ALBEF and TCL.

## Installation

```bash
cd RAO
conda create -n rao python=3.9 -y
conda activate rao
python -m pip install -r requirements.txt
```

Use a CUDA GPU. The scripts use `python` from the active environment; set `PYTHON` to select another interpreter.

## Prepare datasets and models

Download [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/) and place the images in `data/flickr30k/flickr30k-images/`. The test annotation is provided in `data/flickr30k/flickr30k_test.json`, with matching evaluation indices in `data/eval_indices/`. Paths are configured in `paths.py`.

Download the ViT-B/16 and RN101 checkpoints from [OpenAI CLIP](https://github.com/openai/CLIP) and save them as `weights/clip/ViT-B-16.pt` and `weights/clip/RN101.pt`. The Flickr30K-finetuned ALBEF and TCL checkpoints are available in [VLPTransferAttackCheckpoints](https://huggingface.co/Sensen02/VLPTransferAttackCheckpoints/tree/main); place them at `weights/albef_flickr.pth` and `weights/tcl_flickr.pth`.

Download the [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/main) weights and tokenizer files, including `pytorch_model.bin`, and place them in `weights/bert-base-uncased/`.

Download [counter-fitted-vectors.txt](https://drive.google.com/file/d/14Non5yIDaHPUk2TPIskfE-IZ-NiuYjXW/view) and [mat_sim_0.4.txt](https://drive.google.com/file/d/1Uk_IFWdfTLn3rwAmdbT5iY5luv06LSyl/view), then place both files in `data/text/`.

## Run

CLIP ViT-B/16:

```bash
bash run_retrieval_clip_vit.sh --gpu 0 --seed 42 --batch-size 16 --tf32 on
```

CLIP RN101:

```bash
bash run_retrieval_clip_rn101.sh --gpu 1 --seed 42 --batch-size 16 --tf32 on
```

ALBEF:

```bash
bash run_retrieval_albef.sh --gpu 2 --seed 42 --batch-size 16 --tf32 on
```

TCL:

```bash
bash run_retrieval_tcl.sh --gpu 3 --seed 42 --batch-size 16 --tf32 on
```

