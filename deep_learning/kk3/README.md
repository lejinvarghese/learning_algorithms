---
license: mit
task_categories:
- text-generation
- image-to-text
- audio-classification
- zero-shot-classification
language:
- en
multilinguality:
- monolingual
size_categories:
- 100K<n<1M
pretty_name: MultiModal Dataset
tags:
- multimodal
- vision-language
- audio-language
- text
- images
- audio
configs:
- config_name: fineweb
  data_files:
  - split: train
    path: fineweb/train-*.parquet
  - split: valid
    path: fineweb/valid-*.parquet
  - split: test
    path: fineweb/test-*.parquet
- config_name: coco
  data_files:
  - split: train
    path: coco/train-*.parquet
  - split: valid
    path: coco/valid-*.parquet
  - split: test
    path: coco/test-*.parquet
- config_name: audioset
  data_files:
  - split: train
    path: audioset/train-*.parquet
  - split: valid
    path: audioset/valid-*.parquet
  - split: test
    path: audioset/test-*.parquet
---

# Dataset Card for MultiModal Dataset

## Table of Contents
1. [Dataset Card for MultiModal Dataset](#dataset-card-for-multimodal-dataset)
   1. [Table of Contents](#table-of-contents)
   2. [Dataset Description](#dataset-description)
      1. [Dataset Summary](#dataset-summary)
      2. [Supported Tasks](#supported-tasks)
      3. [Languages](#languages)
   3. [Dataset Structure](#dataset-structure)
      1. [Data Instances](#data-instances)
      2. [Data Fields](#data-fields)
         1. [FineWeb Subset](#fineweb-subset)
         2. [COCO Subset](#coco-subset)
         3. [AudioSet Subset](#audioset-subset)
      3. [Data Splits](#data-splits)
   4. [Dataset Creation](#dataset-creation)
      1. [Curation Rationale](#curation-rationale)
      2. [Source Data](#source-data)
         1. [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
         2. [Who are the source language producers?](#who-are-the-source-language-producers)
      3. [Annotations](#annotations)
         1. [Annotation process](#annotation-process)
         2. [Who are the annotators?](#who-are-the-annotators)
      4. [Personal and Sensitive Information](#personal-and-sensitive-information)
   5. [Considerations for Using the Data](#considerations-for-using-the-data)
      1. [Social Impact of Dataset](#social-impact-of-dataset)
      2. [Discussion of Biases](#discussion-of-biases)
      3. [Other Known Limitations](#other-known-limitations)
   6. [Additional Information](#additional-information)
      1. [Dataset Curators](#dataset-curators)
      2. [Licensing Information](#licensing-information)
      3. [Citation Information](#citation-information)
      4. [Contributions](#contributions)
   7. [Quick Start](#quick-start)

## Dataset Description

### Dataset Summary

MultiModal Dataset is a curated collection of **85,000 samples** spanning three modalities: text, images, and audio. It combines high-quality web content, image-caption pairs from COCO 2017, and audio samples from AudioSet to enable comprehensive multimodal model training and evaluation.

The dataset is organized into three subsets:
- **fineweb**: 37,500 high-quality web text samples (>8,192 tokens each)
- **coco**: 37,500 image-caption pairs from COCO 2017 (512x512 resolution)
- **audioset**: 10,000 audio samples with human-annotated labels

Text and image subsets follow an 80/13/7 train/validation/test split. AudioSet uses 80/15/5 split (8K train, 1.5K valid, 500 test).

### Supported Tasks

- **Text Generation**: Long-form text generation using the fineweb subset
- **Image Captioning**: Generate descriptions for images using the coco subset
- **Audio Classification**: Classify audio events using the audioset subset
- **Multimodal Pretraining**: Train models that understand multiple modalities
- **Cross-Modal Retrieval**: Retrieve relevant content across modalities
- **Zero-Shot Classification**: Evaluate zero-shot capabilities across modalities

### Languages

English (en)

## Dataset Structure

### Data Instances

**FineWeb (text) example:**
```json
{
  "id": 42,
  "text": "The history of artificial intelligence begins with ancient myths and stories of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe human thinking as a symbolic system...",
  "modality": "text"
}
```

**COCO (image+text) example:**
```json
{
  "id": 156,
  "text": "A person riding a bicycle on a city street during sunset",
  "image": "<PIL.Image.Image image mode=RGB size=512x512>",
  "modality": "image"
}
```

**AudioSet (audio+text) example:**
```json
{
  "id": 89,
  "text": "Speech, Music, Piano",
  "audio": {
    "array": [0.001, -0.002, 0.003, ...],
    "sampling_rate": 16000
  },
  "modality": "audio"
}
```

### Data Fields

#### FineWeb Subset

| Field | Type | Description |
|-------|------|-------------|
| id | int | Unique sample identifier |
| text | string | High-quality web content (>8,192 tokens) |
| modality | string | Always "text" |

#### COCO Subset

| Field | Type | Description |
|-------|------|-------------|
| id | int | Unique sample identifier |
| text | string | Human-written image caption |
| image | PIL.Image | RGB image at 512x512 resolution |
| modality | string | Always "image" |

#### AudioSet Subset

| Field | Type | Description |
|-------|------|-------------|
| id | int | Unique sample identifier |
| text | string | Comma-separated human-annotated labels |
| audio | dict | Audio data with 'array' (float32 numpy array) and 'sampling_rate' (16000 Hz) |
| modality | string | Always "audio" |

### Data Splits

Text and image subsets use identical 80/13/7 split ratios. AudioSet uses 80/15/5:

| Subset | Train | Validation | Test | Total |
|--------|-------|------------|------|-------|
| fineweb | 30,000 | 5,000 | 2,500 | 37,500 |
| coco | 30,000 | 5,000 | 2,500 | 37,500 |
| audioset | 8,000 | 1,500 | 500 | 10,000 |
| **TOTAL** | **68,000** | **11,500** | **5,500** | **85,000** |

**Split methodology**: 
- **fineweb/coco**: Deterministic index-based partitioning (80% train, 13% valid, 7% test)
- **audioset**: Uses AudioSet's built-in splits (bal_train → train, eval → valid/test)
- Test: Final 7% of samples (indices 0.93×N to N)

This ensures reproducible splits across different loading sessions.

## Dataset Creation

### Curation Rationale

This dataset was created to address the need for a unified multimodal benchmark that:

1. **Provides high-quality data across modalities**: Each subset is sourced from established, well-curated datasets
2. **Maintains consistent splits**: Identical split ratios enable fair cross-modal comparison
3. **Scales appropriately**: 30K-37.5K samples per modality balances diversity with computational feasibility
4. **Focuses on quality over quantity**: FineWeb filters for long-form content (>8K tokens), COCO provides human-verified captions, AudioSet uses expert-annotated labels

### Source Data

#### Initial Data Collection and Normalization

**FineWeb (Text)**
- **Source**: [NoahEJ/fineweb-sample-100BT_over-8192-tokens](https://huggingface.co/datasets/NoahEJ/fineweb-sample-100BT_over-8192-tokens)
- **Original Size**: 100 billion tokens
- **Filtering**: Only documents with >8,192 tokens
- **Rationale**: Long-form content for training models on extended context

**COCO 2017 (Images)**
- **Source**: [wangherr/coco2017_train_512x_image_caption_depth](https://huggingface.co/datasets/wangherr/coco2017_train_512x_image_caption_depth)
- **Original Size**: 118,287 training images
- **Preprocessing**: Resized to 512×512, depth information available
- **Rationale**: Standardized resolution for efficient training, human-verified captions

**AudioSet (Audio)**
- **Source**: [agkphysics/AudioSet](https://huggingface.co/datasets/agkphysics/AudioSet) (config: "full", splits: bal_train/eval)
- **Original Size**: 2 million 10-second clips
- **Processing**: 
  - Audio decoded using torchcodec 0.9.1 (PyTorch 2.9+cu126)
  - FFmpeg 7.0.2 backend for audio file decoding
  - Output: float32 numpy arrays at 16kHz sampling rate
  - Labels converted from list to comma-separated string
- **Rationale**: Diverse sound events with expert human annotations

#### Who are the source language producers?

- **FineWeb**: Web content creators (blog posts, articles, documentation)
- **COCO**: Human annotators hired through crowdsourcing platforms
- **AudioSet**: Ontology developed by Google Research, labels verified by human experts

### Annotations

#### Annotation process

- **FineWeb**: No additional annotation (uses original web content)
- **COCO**: Professional annotators wrote 5 captions per image, this dataset uses 1 caption per image
- **AudioSet**: Trained annotators labeled audio clips following a hierarchical ontology of 632 sound event classes

#### Who are the annotators?

- **COCO**: Crowdworkers via Amazon Mechanical Turk
- **AudioSet**: Expert annotators trained on the AudioSet ontology

### Personal and Sensitive Information

- **FineWeb**: May contain publicly available personal information from web sources
- **COCO**: Images sourced from Flickr; may contain people but no identifying metadata is included
- **AudioSet**: Audio clips from YouTube; may contain speech but no transcriptions or speaker identities

Users should be aware of potential biases and sensitive content when deploying models trained on this data.

## Considerations for Using the Data

### Social Impact of Dataset

**Positive Impacts:**
- Enables research in multimodal AI and cross-modal understanding
- Provides standardized benchmarks for reproducible research
- Supports development of assistive technologies (image captioning for visually impaired, audio classification for hearing impaired)

**Potential Risks:**
- Models trained on web data (FineWeb) may reproduce biases present in internet content
- COCO images may not represent global diversity equitably
- AudioSet may over-represent certain acoustic environments

### Discussion of Biases

- **Geographic bias**: COCO images predominantly from North America and Europe
- **Language bias**: All text in English limits multilingual applicability
- **Domain bias**: FineWeb reflects biases in web content (tech-heavy, Western-centric)
- **Acoustic bias**: AudioSet may over-represent Western musical genres and urban soundscapes

Users should evaluate fairness and bias when deploying models trained on this dataset.

### Other Known Limitations

- **Fixed splits**: Test set may become saturated if widely used for benchmarking
- **Resolution limit**: COCO images fixed at 512×512 (original COCO is variable resolution)
- **Audio duration**: AudioSet clips are typically ~10 seconds (may not represent longer acoustic events)
- **Text length**: FineWeb samples vary in length despite >8K token minimum

## Additional Information

### Dataset Curators

This dataset was curated and compiled by lv12 using memory-efficient data processing pipelines.

### Licensing Information

**Overall License**: MIT License

**Source Dataset Licenses**:
- **FineWeb**: [ODC-By 1.0 License](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- **COCO 2017**: [Creative Commons Attribution 4.0 International License](https://cocodataset.org/#termsofuse)
- **AudioSet**: [Creative Commons Attribution 4.0 International License](https://research.google.com/audioset/download.html)

Users must comply with all source dataset licenses when using this collection.

### Citation Information

If you use this dataset, please cite the original source datasets:

```bibtex
@misc{fineweb2024,
  title={FineWeb: decanting the web for the finest text data at scale},
  author={Penedo, Guilherme and Cappelli, Alessandro and Cojocaru, Liviu and Alobeidli, Hynek and Pannier, Baptiste and Almazrouei, Ebtesam and Launay, Julien},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/HuggingFaceFW/fineweb}
}

@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={Computer Vision--ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13},
  pages={740--755},
  year={2014},
  organization={Springer}
}

@inproceedings{gemmeke2017audio,
  title={Audio set: An ontology and human-labeled dataset for audio events},
  author={Gemmeke, Jort F and Ellis, Daniel PW and Freedman, Dylan and Jansen, Aren and Lawrence, Wade and Moore, R Channing and Plakal, Manoj and Ritter, Marvin},
  booktitle={2017 IEEE international conference on acoustics, speech and signal processing (ICASSP)},
  pages={776--780},
  year={2017},
  organization={IEEE}
}
```

### Contributions

Thanks to the creators of FineWeb, COCO, and AudioSet for making their datasets publicly available.

---

## Quick Start

```python
from datasets import load_dataset

# Load a specific subset and split
ds = load_dataset("lv12/MultiModalDataset", "coco", split="train")

# Iterate through samples
for sample in ds.take(5):
    print(f"ID: {sample['id']}")
    print(f"Text: {sample['text']}")
    print(f"Modality: {sample['modality']}")
    print("---")
```

For more examples, see the [Dataset Structure](#dataset-structure) section.
