import glob
import os
import argparse
import json
import torch
import librosa
from models.stfts import mag_phase_stft, mag_phase_istft
#from datasets.dataset import mag_pha_stft, mag_pha_istft
from models.generator import MambAttentionSEUNet
import soundfile as sf
import time
from tqdm import tqdm

from utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed,
    print_gpu_info, log_model_info, initialize_process_group,
)

h = None
device = None

# handle audio slicing
def process_audio_segment(noisy_wav, model, device, n_fft, hop_size, win_size, compress_factor, sampling_rate, segment_size):
    segment_size = segment_size
    n_fft = n_fft
    hop_size = hop_size
    win_size = win_size
    compress_factor = compress_factor
    sampling_rate = sampling_rate

    norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
    noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
    orig_size = noisy_wav.size(1)

    # whether zeros need to be padded
    if noisy_wav.size(1) >= segment_size:
        num_segments = noisy_wav.size(1) // segment_size
        last_segment_size = noisy_wav.size(1) % segment_size
        if last_segment_size > 0:
            last_segment = noisy_wav[:, -segment_size:]
            noisy_wav = noisy_wav[:, :-last_segment_size]
            segments = torch.split(noisy_wav, segment_size, dim=1)
            segments = list(segments)
            segments.append(last_segment)
            reshapelast=1
        else:
            segments = torch.split(noisy_wav, segment_size, dim=1)
            reshapelast = 0

    else:
        # padding
        padded_zeros = torch.zeros(1, segment_size - noisy_wav.size(1)).to(device)
        noisy_wav = torch.cat((noisy_wav, padded_zeros), dim=1)
        segments = [noisy_wav]
        reshapelast = 0

    processed_segments = []

    for i, segment in enumerate(segments):

        noisy_amp, noisy_pha, noisy_com = mag_phase_stft(segment, n_fft, hop_size, win_size, compress_factor)
        amp_g, pha_g, com_g = model(noisy_amp.to(device, non_blocking=True), noisy_pha.to(device, non_blocking=True))
        audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
        audio_g = audio_g / norm_factor
        audio_g = audio_g.squeeze()
        if reshapelast == 1 and i == len(segments) - 2:
            audio_g = audio_g[ :-(segment_size-last_segment_size)]

        processed_segments.append(audio_g)

    processed_audio = torch.cat(processed_segments, dim=-1)
    #print(processed_audio.size())

    processed_audio = processed_audio[:orig_size]
    #print(processed_audio.size())
    #print(orig_size)

    return processed_audio

def inference(args, device):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']
    segment_size = cfg['training_cfg']['segment_size']

    model = MambAttentionSEUNet(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])

    os.makedirs(args.output_folder, exist_ok=True)

    model.eval()
    pbar = tqdm(total=len(os.listdir( args.input_folder )), desc=f"Running inference", unit="files", leave=False)  # Init pbar
    with torch.no_grad():
        for i, fname in enumerate(os.listdir( args.input_folder )):
            noisy_wav, _ = librosa.load(os.path.join( args.input_folder, fname ), sr=sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)

            output_audio = process_audio_segment(noisy_wav, model, device, n_fft, hop_size, win_size, compress_factor, sampling_rate, segment_size)

            output_file = os.path.join(args.output_folder, fname)
            sf.write(output_file, output_audio.squeeze().cpu().numpy(), sampling_rate, 'PCM_16')
            pbar.update(1)
        pbar.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='VB-DemandEx/noisy_test')
    parser.add_argument('--output_folder', default='results')
    parser.add_argument('--config', default='MambAttention/checkpoints/MambAttention_seed3441_VB-DemandEx.yaml')
    parser.add_argument('--checkpoint_file', default='MambAttention/checkpoints/seed3441.yaml', required=True)
    args = parser.parse_args()

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        #device = torch.device('cpu')
        raise RuntimeError("Currently, CPU mode is not supported.")

    start = time.time()
    inference(args, device)
    print(time.time() - start)

if __name__ == '__main__':
    print("Initializing inference...")
    main()
