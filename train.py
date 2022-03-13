import argparse
import cv2
import os
import random
from glob import glob
from os.path import dirname, join, basename, isfile
import numpy as np
import torch
from syncnet import SyncNet_color as SyncNet
from torch import nn
from torch.nn import functional as F
from torch.utils import data as data_utils
from tqdm import tqdm
import audio
from hparams import hparams, get_image_list
from torch.nn import TripletMarginLoss

parser = argparse.ArgumentParser(description='Code for audio-visual emotion recognition')

parser.add_argument("--data_root", help="Root folder of the preprocessed MEAD dataset", required=True, type=str)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--syncnet_checkpoint_path', help='Load the ', required=True,
                    type=str)

parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)

args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16


class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None:
            return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx: end_idx, :]

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):

        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)  # any random file name from train is chosen
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))  # all the jpg images of the particular folder is stored
            if len(img_names) <= 3 * syncnet_T:
                continue
            img_name = random.choice(img_names)  # any image is chosen from that particular folder
            wrong_img_name = random.choice(img_names)  # random image
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)  # get 5 frames from that video
            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)  # all 5 images are appended

            if not all_read: continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)  # the entire wav file is loaded

                orig_mel = audio.melspectrogram(wav).T  # mel of entire audio
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)  # mel wrt to the frame

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.  # the whole window of 5 frames is then concatenated
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1] // 2:]  # upper half is masked

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y  # x are the frames,
            # m is the mel of true audio and y is either 0 or 1 depending on the true, false condition


triplet_loss = TripletMarginLoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = triplet_loss(d.unsqueeze(1), y)

    return loss


device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step

    while global_epoch < nepochs:
        running_loss = 0.
        print('starting epoch:{}'.format(global_epoch))
        prog_bar = tqdm(enumerate(train_data_loader))

        for step, (frame_seq, mfcc, y) in prog_bar:

            frame_seq = frame_seq.to(device)
            mfcc = mfcc.to(device)
            audio_fv, frame_fv = syncnet(mfcc, frame_seq)

            model.train()
            optimizer.zero_grad()

            y = y.to(device)

            loss = triplet_loss(audio_fv, frame_fv, y)

            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))

        global_epoch += 1


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)