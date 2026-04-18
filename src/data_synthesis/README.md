# Synthetic Misophonia Trigger Soundscape Dataset Construction

This repository describes the procedure used to construct a synthetic dataset for misophonia trigger sound detection.

The dataset was created through the following steps:

1. Convert MATA `.mp4` files into `.wav` audio files
2. Extract foreground candidates using `extract_foreground.py`
3. Extract noise segments using `noise_reduction.py`
4. Screen candidate sounds with YAMNet using `yamnet_screening.py`
5. Manually inspect the audio and retain only the selected samples
6. Split the data using `split_move_by_prefix.py`
7. Prepare background audio
8. Generate synthetic soundscapes using the DCASE Task 4 repository

## 1. Convert MATA videos (`.mp4`) to audio (`.wav`)

First, audio tracks are extracted from the MATA `.mp4` video files and converted into `.wav` format.  
The resulting `.wav` files are used in the subsequent foreground extraction process.

### Input
- `.mp4` files under `MATA/`

### Output
- `.wav` files under `audio/`

### Example command

```bash
mkdir -p audio

find MATA -type f -name "*.mp4" | while read f; do
  base=$(basename "$f" .mp4)
  ffmpeg -i "$f" -vn -ac 1 -ar 16000 "audio/${base}.wav"
done
```

## 2. Foreground extraction (`extract_foreground.py`)

This script extracts foreground event candidates from source audio files and converts them into DESED-style event clips.

In our pipeline, we used this step to segment candidate foreground sounds from the prepared source audio. The script first loads each audio file, trims leading and trailing silence, detects active intervals based on energy, merges nearby intervals, and exports each resulting segment as an individual waveform file. Very short segments shorter than 0.25 s are discarded. For micro-burst classes such as `clock`, `typing`, and `sniffing`, nearby short bursts are additionally packed into a single segment when appropriate.

### Supported target classes

The script uses the following canonical foreground classes:

- `chewing`
- `sniffing`
- `throat_clearing`
- `coughing`
- `clock`
- `breathing`
- `typing`

### Input

The script can read audio from dataset-specific directories or from a custom input directory, depending on how it is configured.

In the dataset-specific mode, it includes lightweight iterators for datasets such as ESC-50, FSD50K, VocalSound, FOAMS, and MATA. In the simple folder-based mode, it recursively scans an input directory and reads audio files whose parent directory names correspond to canonical class labels. Supported extensions are `.wav`, `.mp3`, `.m4a`, `.aac`, `.flac`, and `.ogg`.

### Output

Extracted event clips are saved under class-wise subdirectories:

```bash
<out_root>/<class>/
```

## 3. Noise reduction (`extract_noise.py`)

This script applies batch noise reduction to `.wav` files in a target directory and saves the processed files to an output directory.

In our pipeline, this step was used to reduce stationary background noise in selected foreground recordings before subsequent processing.

### Input

The script reads `.wav` files from the input directory.

### Output

Noise-reduced files are saved to the output directory using the same filenames as the originals.

### Processing method

For each input file, the script performs the following steps:

1. Load the audio file
2. Extract a noise reference segment from the beginning of the recording
3. Apply stationary noise reduction using `noisereduce`
4. Save the processed waveform to the output directory

### Noise reference

The script uses the first 0.5 seconds of each audio file as the noise reference by default.

If the audio is shorter than 0.5 seconds, the entire signal is used as the noise reference instead.

### File format handling

The script searches for both `.wav` and `.WAV` files in the input directory.

### Default parameters

- `INPUT_DIR`: directory containing noisy `.wav` files
- `OUTPUT_DIR`: directory to save denoised files
- `NOISE_DURATION_SEC = 0.5`

### Example usage

```bash
python extract_noise.py

## 3. Candidate screening with YAMNet (`yamnet_screening.py`)

After foreground extraction, we used YAMNet to screen the candidate audio clips.

This script recursively scans the input directory for audio files, runs the official YAMNet model from TensorFlow Hub on each clip, and copies or moves the files into class-wise subdirectories according to the top-1 predicted label. A CSV report containing the predicted label, prediction score, and top-k labels is also generated.

### Input

The script reads audio files recursively from `input_dir`. By default, it processes both `.wav` and `.WAV` files.

### Output

The script creates class-wise subdirectories under `output_dir` and places each file into the directory corresponding to its top-1 predicted label.

A CSV report is also written under `output_dir`.

### Processing steps

For each input file, the script performs the following steps:

1. Read the waveform
2. Convert multi-channel audio to mono if necessary
3. Resample the waveform to 16 kHz if required
4. Normalize the waveform to approximately `[-1, 1]`
5. Run YAMNet inference
6. Average frame-level prediction scores over time
7. Select the top-1 predicted label
8. Copy or move the file into the corresponding class directory
9. Save prediction results to a CSV report

### YAMNet model

We used the official YAMNet model provided via TensorFlow Hub:

`https://tfhub.dev/google/yamnet/1`

### Handling uncertain predictions

If the top-1 prediction score is lower than `min_score`, the file is placed into the directory specified by `uncertain_dir` instead of a predicted class directory.

### Output report

The script writes a CSV report containing:

- source file path
- predicted label
- predicted score
- top-k labels
- top-k scores

### Example usage

```bash
python yamnet_screening.py \
  --input_dir /path/to/input \
  --output_dir /path/to/output \
  --mode copy \
  --top_k 5 \
  --min_score 0.0
```

## 4. Manual verification

After YAMNet-based screening, we manually reviewed the clips that were not classified into the target classes by YAMNet.

During this step, we listened to each candidate clip and excluded only those that were clearly perceived as distinct non-target sounds. Clips that were not clearly identifiable as non-target sounds were retained as candidate foreground samples.

Only the data reflected in `data/jams/` were used in the final dataset construction. No additional clips outside those metadata files were used.

## 5. Build train/eval/test soundbank splits (`build_soundbank_split.py`)

After foreground and background clips were prepared, we organized them into train/eval/test soundbank splits.

This script recursively scans an archive directory, infers the dataset name from the filename prefix, infers the sound domain (`foreground` or `background`) and category from the directory structure, and assigns files to train/eval/test splits while grouping related files by their inferred origin. The grouped splitting is intended to reduce leakage between splits when multiple clips are derived from the same original recording.

By default, the script performs the split independently for each dataset. The resulting files are placed under:

```bash
<out_root>/<split>/soundbank/<domain>/<category>/
```

---

## Input / Output

### Input

The script reads audio files recursively from an archive root directory. Supported extensions include:

- `.wav`
- `.mp3`
- `.flac`
- `.m4a`
- `.aac`
- `.ogg`
- `.wma`

### Output

The output directory is organized as:

```bash
<out_root>/<split>/soundbank/<domain>/<category>/
```

---

### Processing steps

For each file in the archive, the script performs the following steps:

1. Scan the archive recursively for supported audio files
2. Infer the dataset name from the filename prefix
3. Infer the sound domain and category from the directory structure
4. Infer an origin identifier from the filename by omitting selected tail tokens
5. Group files by origin
6. Split grouped origins into train/eval/test according to the specified ratios
7. Place each file into the corresponding soundbank directory
8. Save summary reports for the pre-split and post-split file counts

### Origin grouping

To avoid splitting related clips across different subsets, the script groups files by an inferred origin identifier. The origin is derived from the filename after removing the dataset prefix and omitting selected tail positions. By default, the second-last and third-last tokens are omitted (`--omit_from_end 2,3`).

This heuristic is useful when filenames contain variable segment-specific suffixes such as timestamps or clip indices.

### Split settings

The split ratio can be specified with `--ratios`, whose default value is `70,15,15`.

The split is reproducible via `--seed`. By default, the script performs the split independently for each dataset (`--per_dataset_split`).


---

### Example usage

```bash
python build_soundbank_split.py \
  --src /path/to/archive \
  --out_root /path/to/output \
  --ratios 60,20,20 \
  --seed 42 \
  --dry_run
```

## 6. Background split

Background audio files were either assigned according to the original predefined split, when such split information was available, or divided into train/eval/test subsets using a 6:2:2 ratio otherwise.

## 7. Soundscape synthesis

Finally, synthetic soundscapes were generated using the DCASE20_TASK4 repository.

In our pipeline, the prepared foreground and background soundbank files were used as inputs to the soundscape synthesis procedure provided in DCASE20_TASK4.

For reference, the repository is available at:

- DCASE20_TASK4: `https://github.com/turpaultn/dcase20_task4`