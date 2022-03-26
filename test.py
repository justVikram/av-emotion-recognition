import argparse
import cv2
import os
import random
from glob import glob
from os.path import dirname, join, basename, isfile
import numpy as np
import torch
from models.syncnet import SyncNet_color as SyncNet
from torch import optim
from torch.utils import data as data_utils
from tqdm import tqdm
import audio
from hparams import hparams, get_image_list
from torch.nn import TripletMarginLoss
from models.wav2lip import Wav2Lip as Wav2Lip

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
syncnet_mel_step_size = 13  # TODO: What to select?


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
        i = 0
        label = str()
        query = dict()
        support = dict()
        anchor_window = []
        positive_mel = []
        while len(support.keys()) <= 2:
            while 1:
                idx = random.randint(0, len(self.all_videos) - 1)  # any random folder name from train is chosen
                vid_name = self.all_videos[idx]
                identifiers = vid_name.split('_')
                label = identifiers[2]
                speaker_identity = identifiers[0]

                img_names = list(glob(join(vid_name, '*.jpg')))  # all the jpg images of the particular folder is stored
                if len(img_names) <= 3 * syncnet_T:
                    continue
                anchor_frame = random.choice(img_names)  # any image is chosen from that particular folder

                anchor_frames = self.get_window(anchor_frame)  # get 5 frames from that video
                if anchor_frames is None:
                    continue

                window = []
                all_read = True
                for frame in anchor_frames:
                    img = cv2.imread(frame)
                    if img is None:
                        all_read = False
                        break
                    try:
                        img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                    except Exception as e:
                        all_read = False
                        break
                    window.append(img)  # all 2 images are appended

                if not all_read:
                    continue

                # MFCC from positive audio sample
                try:
                    audio_file = join(vid_name, "audio.wav")
                    wav = audio.load_wav(audio_file, hparams.sample_rate)  # the entire wav file is loaded
                    full_length_mfcc = audio.melspectrogram(wav).T  # mel of entire audio
                except Exception as e:
                    continue

                positive_mel = self.crop_audio_window(full_length_mfcc.copy(), anchor_frame)  # mel wrt to the frame

                if positive_mel.shape[0] != syncnet_mel_step_size:
                    continue

                anchor_window = np.concatenate(window,
                                               axis=2) / 255.  # the whole window of 5 frames is then concatenated
                anchor_window = anchor_window.transpose(2, 0, 1)

                anchor_window = torch.FloatTensor(anchor_window)
                positive_mel = torch.FloatTensor(positive_mel.T).unsqueeze(0)
                break

            if i == 0:
                i = 1
                query.update({
                    label: (anchor_window, positive_mel)
                })

            else:
                support.update({
                    label: (anchor_window, positive_mel)
                })
        return query, support


triplet_loss = TripletMarginLoss()

device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False


def test(device, model, train_data_loader, test_data_loader, optimizer,
         checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step

    while global_epoch < nepochs:
        running_loss = 0.
        print('starting epoch:{}'.format(global_epoch))
        prog_bar = tqdm(enumerate(train_data_loader))

        for step, (anchor_window, positive_mel, negative_mel) in prog_bar:

            anchor_window = anchor_window.to(device)
            positive_mel = positive_mel.to(device)
            negative_mel = negative_mel.to(device)
            positive_audio_fv, frame_fv = syncnet(positive_mel, anchor_window)
            negative_audio_fv, _ = syncnet(negative_mel, anchor_window)

            frame_fv.requires_grad_(True)

            model.test()
            optimizer.zero_grad()

            loss = triplet_loss(frame_fv, positive_audio_fv, negative_audio_fv)

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


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 700
    print('Evaluating for {} steps'.format(eval_steps))
    sync_losses, recon_losses = [], []
    step = 0
    trip_loss_history = []
    while 1:
        for support, query in test_data_loader:
            step += 1
            model.eval()

            anchor_window = anchor_window.to(device)
            positive_mel = positive_mel.to(device)
            negative_mel = negative_mel.to(device)
            positive_audio_fv, frame_fv = model(positive_mel, anchor_window)
            negative_audio_fv, _ = model(negative_mel, anchor_window)

            loss = triplet_loss(frame_fv, positive_audio_fv, negative_audio_fv)
            trip_loss_history.append(loss)

            if step > eval_steps:
                averaged_trip_loss = sum(trip_loss_history) / len(trip_loss_history)

                print('Trip loss: {}'.format(averaged_trip_loss))

                return averaged_trip_loss


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


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model


if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir

    # Dataset and Data-loader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=4)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = Wav2Lip().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)

    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Train!
    test(device, model, train_data_loader, test_data_loader, optimizer,
         checkpoint_dir=checkpoint_dir,
         checkpoint_interval=hparams.checkpoint_interval,
         nepochs=hparams.nepochs)
