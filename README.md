<h2 align="center">X-AVDT: Audio-Visual Cross-Attention for Robust Deepfake Detection</h2>
<div align="center"> 
  <a href="https://youngseo0526.github.io/" target="_blank">Youngseo Kim</a> · 
  <a href="https://kwanyun.github.io/" target="_blank">Kwan Yun</a> · 
  <a href="https://seokhyeonhong.github.io/" target="_blank">Seokhyeon Hong</a> · 
  <a href="https://chacorp.github.io/sihuncha/" target="_blank">Sihun Cha</a> · 
  <a href="https://sj0414.github.io/colette-koo/" target="_blank">Colette Suhjung Koo</a> ·
  <a href="https://vml.kaist.ac.kr/main/people/person/1" target="_blank">Junyong Noh</a>
</div>
<p align="center"> 
  <b>Visual Media Lab @ KAIST</b><br/>CVPR 2026
</p>

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://youngseo0526.github.io/X-AVDT/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2603.08483)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/zaqxsw0526/MMDF)

This repository contains the official implementation of the paper "X-AVDT: Audio-Visual Cross-Attention for Robust Deepfake Detection".

![](docs/static/images/x-avdt_poster_cvpr26.png)

TL;DR : audio-visual cross-attention with diffusion inversion features, a more robust detector against unseen deepfake generators.

## Installation

```bash
conda create -n x-avdt python=3.10 -y
conda activate x-avdt
pip install -r requirements.txt
```

`ffmpeg` is required for video preprocessing.
## Download Pretrained Models

The Hallo feature extraction uses the original [Hallo](https://github.com/fudan-generative-vision/hallo#%EF%B8%8F%EF%B8%8F-usage) repository. Pretrained weights are placed under `hallo/pretrained_models`. 
```text
hallo/pretrained_models/
```

The easiest setup is to clone the official pretrained model bundle from Hugging Face:

```bash
cd hallo
git lfs install
git clone https://huggingface.co/fudan-generative-ai/hallo pretrained_models
cd ..
```

If downloading files manually, organize them as follows:

```text
hallo/pretrained_models/
  audio_separator/
    download_checks.json
    mdx_model_data.json
    vr_model_data.json
    Kim_Vocal_2.onnx
  face_analysis/
    models/
      face_landmarker_v2_with_blendshapes.task
      1k3d68.onnx
      2d106det.onnx
      genderage.onnx
      glintr100.onnx
      scrfd_10g_bnkps.onnx
  hallo/
    net.pth
  motion_module/
    mm_sd_v15_v2.ckpt
  sd-vae-ft-mse/
    config.json
    diffusion_pytorch_model.safetensors
  stable-diffusion-v1-5/
    unet/
      config.json
      diffusion_pytorch_model.safetensors
  wav2vec/
    wav2vec2-base-960h/
      config.json
      feature_extractor_config.json
      model.safetensors
      preprocessor_config.json
      special_tokens_map.json
      tokenizer_config.json
      vocab.json
```

These paths match `hallo/configs/inference/default.yaml`.

## Dataset
The MMDF can be downloaded from [Hugging Face](https://huggingface.co/datasets/zaqxsw0526/MMDF).

After feature extraction, training expects this layout for both real and fake roots:

```text
<root>/<split>/<label>/<model_id>/<clip_id>/
  original/*.pt
  inverted/*.pt
  reconstructed/*.pt
  residual/*.pt
  attn_feat/*.pt
```

## Feature Extraction

If your inputs are raw videos, first convert them into frame folders and wav files:

```bash
python hallo/preprocess_videos.py extract-frames \
  --video_dir /path/to/videos \
  --frames_dir /path/to/frames \
  --duration 5 \
  --fps 25 \
  --size 512 512
```

Then run Hallo feature extraction. This produces whole-clip outputs such as `original.mp4`, `inverted.mp4`, `reconstructed.mp4`, `residual.mp4`, `attn_map.mp4`, and `attn_feat.pt` for each clip:

```bash
python hallo/extract_features.py \
  --frames_dir /path/to/frames \
  --output_dir /path/to/hallo_features
```

Finally, pack the whole-clip feature outputs into the training layout. This step slices each clip into 16-frame `.pt` chunks:

```bash
python hallo/preprocess_videos.py pack-features \
  --feature_dir /path/to/hallo_features \
  --output_dir /path/to/pt \
```

## Training

```bash
python train/train.py --data_dir /path/to/pt/ 
```

## Evaluation
Download the pretrained X-AVDT detector weights from the following [link](https://drive.google.com/file/d/1O5Xov2UQMIApzSAf63aJ80owJjunkxS7/view?usp=sharing). Then run the evaluation scripts:
```bash
python train/evaluate.py --data_dir /path/to/pt/ --ckpt results/x_avdt/model_best.pt 
```


## Citation

```bibtex
@InProceedings{Kim_2026_CVPR,
    author    = {Kim, Youngseo and Yun, Kwan and Hong, Seokhyeon and Cha, Sihun and Koo, Colette Suhjung and Noh, Junyong},
    title     = {X-AVDT: Audio-Visual Cross-Attention for Robust Deepfake Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {4403-4414}
}
```
