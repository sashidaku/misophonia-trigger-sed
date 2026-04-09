Attribution information for source materials is provided in the `attributions/` directory, organized by dataset.
This release does not redistribute the original source audio.

## What this repository provides

This repository provides:
- source code for dataset synthesis and experiments
- configuration files
- JAMS metadata files required to reconstruct the synthetic soundscapes
- attribution / license metadata for the source clips

This repository does **not** redistribute any original audio or video files from third-party datasets.

## Important licensing notice

The synthetic soundscapes used in this study were constructed from multiple third-party datasets with different upstream license terms.

Examples include:
- MATA: CC BY-NC 4.0
- FOAMS (processed audio release): CC BY 4.0
- VocalSound: CC BY-SA 4.0
- ESC-50: CC BY-NC, with ESC-10 clips under CC BY
- FSD50K: clip-level Creative Commons licenses (e.g., CC0, CC BY, CC BY-NC, CC Sampling+)

Accordingly, this repository does **not** claim to relicense or redistribute the original third-party media.
Users must obtain the source datasets separately from their official distribution channels and comply with the applicable upstream license terms for each dataset or clip.

## About the JAMS files

The JAMS files distributed in this repository are metadata files created for reproducibility.
They describe how the synthetic soundscapes were generated, including source identifiers, event timing, and processing history.
They do not include the original source media.

The inclusion of source identifiers, attribution fields, and upstream license information is provided solely to support reproducibility and proper attribution.
Rights in the original third-party media remain with their respective rights holders.

## Modifications applied to source material

Depending on the source dataset and clip, preprocessing may include:
- format conversion (e.g., mp4 to wav)
- cropping / segmentation
- resampling (e.g., to 16 kHz)
- noise reduction

Please refer to the accompanying metadata files for per-clip processing history.

## License for this repository

Unless otherwise stated:
- the **code** in this repository is licensed under [MIT/Apache-2.0/etc.]
- third-party source data are **not** included in this license grant
- metadata entries describing third-party materials do **not** grant rights to the underlying source media

## How to reconstruct the dataset

To reconstruct the synthetic dataset:
1. Obtain the required source datasets from their official providers.
2. Verify that your intended use complies with the applicable upstream license terms.
3. Place the source data locally in the expected directory structure.
4. Run the reconstruction scripts using the released JAMS metadata.

## Citation

For details of the preprocessing pipeline, dataset synthesis procedure, and reconstruction workflow, please refer to this repository:
https://github.com/sashidaku/misophonia-trigger-sed

## Citation

If you use this repository, the released JAMS metadata, or the reconstructed dataset in your research, please cite our paper

Please also cite the original source datasets and comply with their respective attribution and license requirements.
Per-dataset and per-clip attribution information is provided in the `attributions/` directory.

