# Misophonia Trigger Sound Detection

This repository provides code for misophonia trigger sound detection on synthetic soundscapes, corresponding to the IJCNN'26 paper “[Misophonia Trigger Sound Detection on Synthetic Soundscapes Using a Hybrid Model with a Frozen Pre-Trained CNN and a Time-Series Module](https://arxiv.org/abs/2602.06271).”


Misophonia is a disorder characterized by decreased tolerance to specific everyday sounds or associated cues, often referred to as *trigger sounds*. In this work, we study sound event detection (SED) for misophonia-related trigger sounds as a step toward future assistive on-device applications.

All models in this repository share the same overall framework for sound event detection: a **frozen pre-trained frame-wise CNN backbone**, a **temporal modeling module**, and a **frame-wise linear readout** for multi-label prediction.

The CNN backbone is **fmn10**　[1], which is used as a fixed feature extractor.  
Given an input audio clip, fmn10 produces frame-level acoustic embeddings, and these embeddings are then processed by one of the following temporal modules:

- **Linear**: a frame-wise linear baseline without temporal recurrence
- **GRU**: a recurrent temporal model for sequential modeling
- **LSTM**: a recurrent temporal model with explicit memory cells
- **ESN**: an Echo State Network-based temporal model with a fixed reservoir and trainable readout


## Pretrained CNN 
This repository uses the publicly available **fmn10** pretrained checkpoint from the official EfficientSED implementation [1].
Please download the checkpoint from the official EfficientSED source and place it in the `resources/` directory, following the original EfficientSED setup.

## Target Classes

The current dataset includes the following trigger-related classes:

- breathing
- coughing
- eating
- sniffing
- throat clearing
- typing
- clock ticking

## Source Datasets
Audio clips were collected from the following public datasets and resources:

## Source Datasets

Audio clips were collected from the following public datasets and resources:

### Foreground sound sources
- [FOAMS](https://zenodo.org/records/7109069)
- [MATA](https://github.com/Svetlana-Shinkareva/MATA)
- [FSD50K](https://zenodo.org/records/4060432)
- [ESC-50](https://github.com/karolpiczak/ESC-50)
- [VocalSound](https://github.com/YuanGongND/vocalsound)

### Background sound sources
- [DCASE 2018 Task 5 (derived from SINS)](https://zenodo.org/records/1247102)
- [TUT Acoustic Scenes 2017](https://zenodo.org/records/400515)
- [MUSAN](https://www.openslr.org/17/)

### jams files to synthesize misophonia trigger sound detaset
This repository includes scripts for generating synthetic soundscapes from JAMS metadata.

Please note that the audio files used in our setup are organized with filenames in the following format:

`<dataset_name>_<file_id>_<start_time>_<end_time>_sr<sampling_rate>.wav`

As a result, the provided JAMS files are not always directly compatible with the original filenames from each source dataset.
Users may need to prepare their own filename mapping or metadata alignment to reproduce the generation pipeline in their local environment.

More detailed instructions will be added in the future.


## References
[1] T. Morocutti, F. Schmid, J. Greif, F. Foscarin and G. Widmer, "Exploring Performance-Complexity Trade-Offs in Sound Event Detection Models," in Proc. 33rd European Signal Processing Conference (EUSIPCO), Palermo, Italy, 2025, pp. 126-130, doi: 10.23919/EUSIPCO63237.2025.11226457.




## Repository Structure

```text
misophonia-trigger-sed/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── train_multiclass.yaml
│   ├── eval_multiclass.yaml
│   ├── train_fewshot.yaml
│   └── eval_fewshot.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── run_fewshot.py
│   ├── build_dataset.py
│   └── make_jams.py
├── src/
│   └── misophonia_trigger_sed/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       └── utils/
├── tests/
└── outputs/
```

## Acknowledgements

This repository uses **fmn10** [1] as the frozen CNN backbone.

Parts of the implementation were adapted from the official EfficientSED codebase:
- [EfficientSED GitHub repository](URL)

We thank the original authors for publicly releasing their implementation.


