import glob
import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from tqdm import tqdm

INPUT_DIR = "/srv/datasets/clock"

OUTPUT_DIR = "/srv/datasets/clock/noise_reduced"

NOISE_DURATION_SEC = 0.5

def batch_noise_reduction(input_folder, output_folder, noise_duration):
    print(f"starting noise reduction...")
    print(f"input directory: {input_folder}")
    print(f"output directory: {output_folder}")

    os.makedirs(output_folder, exist_ok=True)

    file_list = glob.glob(os.path.join(input_folder, "*.wav"))
    file_list.extend(glob.glob(os.path.join(input_folder, "*.WAV")))

    if not file_list:
        print(f"Warning: No .wav files found in {input_folder}.")
        return

    print(f"total {len(file_list)} files to process.")

    for file_path in tqdm(file_list, desc="Processing files"):
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, filename)

        try:
            data, sr = librosa.load(file_path, sr=None)

            noise_sample_len = int(noise_duration * sr)
            if len(data) < noise_sample_len:
                noise_clip = data
            else:
                noise_clip = data[:noise_sample_len]

            reduced_noise_data = nr.reduce_noise(
                y=data,
                sr = sr,
                y_noise=noise_clip,
                stationary=True,
            )

            sf.write(output_path, reduced_noise_data, sr)

        except Exception as e:
            print(f"Error: {filename}: {e}")

    print(f"✅ Done check the output folder: {output_folder}")

if __name__ == "__main__":
    if INPUT_DIR == "./your_noisy_files_folder" or OUTPUT_DIR == "./your_clean_files_folder":
        print("Error: Please edit the `INPUT_DIR` and `OUTPUT_DIR` variables at the top of the script.")
    else:
        batch_noise_reduction(INPUT_DIR, OUTPUT_DIR, NOISE_DURATION_SEC)