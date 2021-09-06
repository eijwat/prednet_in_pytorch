import os
import argparse
import numpy as np
import cv2
import librosa
import librosa.display
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt

usage = 'Usage: python {} INPUT_FILE [--dir <directory>] [--help]'.format(__file__)
parser = argparse.ArgumentParser(description='This script is to generate images from a wave file.',
                                 usage=usage)
subparsers = parser.add_subparsers(dest='command')
wave_to_db = subparsers.add_parser('wave_to_db', help='see `add -h`')
wave_to_db.add_argument('input_sound', action='store', nargs=None,
                        type=str, help='Input sound file.')
wave_to_db.add_argument('-o', '--outdir', action='store', default="data",
                        type=str, help='Output directory.')
wave_to_db.add_argument('-sr', '--sampling_rate', action='store', default=44100,
                        type=int, help='Sampling rate.')
wave_to_db.add_argument('-t', '--time_size', action='store',
                        default=160, type=int, help='Length of sound time sequence.')
wave_to_db.add_argument('-s', '--skip_size', action='store',
                        default=160, type=int, help='Skip length of time sequence.')
wave_to_db.add_argument('-w', '--window', action='store',
                        default=1024, type=int, help='FFT window size.')
wave_to_db.add_argument('-l', '--hop_length', action='store',
                        default=1024 // 4, type=int, help='Hop length.')
wave_to_db.add_argument('-f', '--frq_size', action='store',
                        default=512, type=int, help='Frequency size.')
wave_to_db.add_argument('--with_image', action='store_true',
                        help='Save with spectrum image.')
wave_to_db.add_argument('--range', action='store',
                        default="-80,10", help='Data range for normalization.')

db_to_wave = subparsers.add_parser('db_to_wave', help='see `add -h`')
db_to_wave.add_argument('-sr', '--sampling_rate', action='store', default=44100,
                        type=int, help='Sampling rate.')
db_to_wave.add_argument('outputs_dir', action='store', nargs=None,
                        type=str, help='Outputs directory.')
db_to_wave.add_argument('-w', '--window', action='store',
                        default=1024, type=int, help='FFT window size.')
db_to_wave.add_argument('-l', '--hop_length', action='store',
                        default=1024 // 4, type=int, help='Hop length.')
db_to_wave.add_argument('--with_image', action='store_true',
                        help='Save with spectrum image.')
db_to_wave.add_argument('--range', action='store',
                        default="-80,10", help='Data range for normalization.')
db_to_wave.add_argument('--merge', action='store_true', help='Merge wave datum.')
db_to_wave.add_argument('-s', '--skip_size', action='store',
                        default=160, type=int, help='Skip length of time sequence.')
args = parser.parse_args()


def wave_to_db():
    base_length = int(5e7)
    args.range = args.range.split(',')
    for i in range(len(args.range)):
        args.range[i] = int(args.range[i])
    basename = os.path.splitext(os.path.basename(args.input_sound))[0]
    print("Load data...")
    x, _ = librosa.load(args.input_sound, sr=args.sampling_rate)
    x = librosa.to_mono(x)
    if x.shape[0] > base_length:
        base_length = base_length // args.hop_length * args.hop_length
        xlist = [x[i:min(i + base_length + args.window, x.shape[0])] for i in range(0, x.shape[0], base_length)]
        if xlist[-1].shape[0] < args.window:
            xlist = xlist[:-1]
    else:
        xlist = [x]

    flist = []
    cnt = 0
    for x in tqdm(xlist):
        stft = librosa.stft(x, n_fft=args.window, hop_length=args.hop_length, center=False)  # stft
        stft_v = np.stack([np.real(stft), np.imag(stft)], axis=-1).astype(np.float32)  # complexes to 2D vecotors
        stft_v = cv2.resize(stft_v, (stft_v.shape[1], args.frq_size), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)  # resize
        stft_v = librosa.power_to_db(stft_v)  # power to db
        stft_v = (stft_v - min(args.range)) / abs(args.range[0] - args.range[1])  # normalize
        for i in range(0, stft.shape[1] - args.time_size, args.skip_size):
            fname = os.path.join(args.outdir, basename + "_%012d.npy" % cnt)
            np.save(fname, stft_v[..., i:(i + args.time_size)])
            flist.append(fname)
            if args.with_image:
                librosa.display.specshow(librosa.power_to_db(stft[:, i:(i + args.time_size)]),
                                         y_axis='log', x_axis='time', sr=args.sampling_rate)
                plt.colorbar(format='%+2.0f dB')
                plt.savefig(os.path.join(args.outdir, basename + "_%012d.png" % cnt))
                plt.close()
            cnt += 1
    with open(os.path.join(args.outdir, "train_list.txt"), 'w') as f:
        for fname in flist:
            f.write("%s\n" % fname)


def db_to_wave():
    args.range = args.range.split(',')
    for i in range(len(args.range)):
        args.range[i] = int(args.range[i])

    filelist = [f for f in os.listdir(args.outputs_dir) if os.path.splitext(f)[-1] == ".npy"]
    filelist.sort()
    merged_stft = []
    merged_x = []
    for i, f in enumerate(tqdm(filelist)):
        basename = os.path.splitext(os.path.basename(f))[0]
        stft = np.load(os.path.join(args.outputs_dir, f))
        stft = stft.transpose(1, 2, 0)
        stft = stft * abs(args.range[0] - args.range[1]) + min(args.range)  # denormalize
        stft = librosa.db_to_power(stft)  # db to power
        stft = cv2.resize(stft, (stft.shape[1], args.window // 2  + 1), interpolation=cv2.INTER_LINEAR)  # resize
        stft = np.vectorize(complex)(stft[..., 0], stft[..., 1])  # 2D vectors to complexes
        x = librosa.istft(stft, hop_length=args.hop_length, win_length=args.window, center=False)  # inverse stft
        sf.write(os.path.join(args.outputs_dir, os.path.splitext(f)[0] + ".wav"), x, args.sampling_rate)
        if args.with_image:
            librosa.display.specshow(librosa.power_to_db(stft),
                                     y_axis='log', x_axis='time', sr=args.sampling_rate)
            plt.colorbar(format='%+2.0f dB')
            plt.savefig(os.path.join(args.outputs_dir, basename + ".png"))
            plt.close()
        if args.merge:
            idx = idx if i == len(filelist) - 1 else min(args.skip_size, stft.shape[1])
            merged_stft.append(stft[:, :idx])
        if args.merge and ((i > 0 and i % int(1e5) == 0) or i == len(filelist) - 1):
            merged_stft = np.concatenate(merged_stft, axis=1)
            merged_x.append(librosa.istft(merged_stft, hop_length=args.hop_length, win_length=args.window, center=False))
            merged_stft = []
    if args.merge:
        merged_x = np.concatenate(merged_x, axis=0)
        sf.write(os.path.join(args.outputs_dir, "merged_out.wav"), merged_x, args.sampling_rate)


if __name__ == "__main__":
    if args.command == "wave_to_db":
        wave_to_db()
    else:
        db_to_wave()
