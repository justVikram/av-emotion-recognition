

# python  test.py --data_root ~/AV/MEAD/ --checkpoint_dir ~/AV/av-emotion-recognition/eval_checkpoints/ --syncnet_checkpoint_path ~/AV/latest.pth
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
from sklearn.metrics.pairwise import cosine_similarity

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
syncnet_mel_step_size = 13


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
            idx = random.randint(0, len(self.all_videos) - 1)  # any random folder name from train is chosen
            vid_name = self.all_videos[idx]
            identifiers = vid_name.split('_')
            label = identifiers[2]
            speaker_identity = identifiers[0]

            negative_idx = random.randint(0, len(self.all_videos) - 1)
            negative_vid_name = self.all_videos[negative_idx]
            negative_identifiers = negative_vid_name.split('_')
            negative_label = negative_identifiers[2]
            negative_speaker_identity = negative_identifiers[0]

            if label == negative_label and speaker_identity != negative_speaker_identity:
                continue

            img_names = list(glob(join(vid_name, '*.jpg')))  # all the jpg images of the particular folder is stored
            if len(img_names) <= 3 * syncnet_T:
                continue
            anchor_frame = random.choice(img_names)  # any image is chosen from that particular folder

            anchor_frames = self.get_window(anchor_frame)  # get 5 frames from that video
            if anchor_frames is None:
                continue

            negative_img_names = list(glob(join(negative_vid_name, '*.jpg')))
            if len(negative_img_names) <= 3 * syncnet_T:
                continue

            # Selecting a random anchor frame from the negative video
            negative_anchor_frame = random.choice(negative_img_names)

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

            # MFCC from negative audio sample
            try:
                negative_audio_file = join(negative_vid_name, "audio.wav")
                negative_wav = audio.load_wav(negative_audio_file, hparams.sample_rate)  # the entire wav file is loaded
                negative_full_length_mfcc = audio.melspectrogram(negative_wav).T
            except Exception as e:
                continue

            negative_mel = self.crop_audio_window(negative_full_length_mfcc.copy(), negative_anchor_frame)

            if negative_mel.shape[0] != syncnet_mel_step_size:
                continue

            # H x W x 3 * T
            anchor_window = np.concatenate(window, axis=2) / 255.  # the whole window of 5 frames is then concatenated
            anchor_window = anchor_window.transpose(2, 0, 1)

            anchor_window = torch.FloatTensor(anchor_window)
            positive_mel = torch.FloatTensor(positive_mel.T).unsqueeze(0)
            negative_mel = torch.FloatTensor(negative_mel.T).unsqueeze(0)

            return anchor_window, positive_mel, negative_mel  # x are the frames,
            # m is the mel of true audio and y is the emotion label


class DatasetTest(Dataset):
    def __init__(self, split):
        super(DatasetTest, self).__init__(split)

    def __getitem__(self, idx):
        query = dict()
        support = dict()
        anchor_window = []
        positive_mel = []
        #print("getitem() was called")
        while len(support.keys()) < 4:
            while 1:
                idx = random.randint(0, len(self.all_videos) - 1)  # any random folder name from test is chosen
                vid_name = self.all_videos[idx]
                identifiers = vid_name.split('_')
                label = identifiers[2]

                speaker_identity = identifiers[0]

                img_names = list(glob(join(vid_name, '*.jpg')))  # all the jpg images of the particular folder is stored
                if len(img_names) <= 3 * syncnet_T:
                    continue
                anchor_frame = random.choice(img_names)  # any image is chosen from that particular folder

                anchor_frames = self.get_window(anchor_frame)  # get 5 frames from that video
                if anchor_frames is None:  # if the video is too short
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
                    window.append(img)  # all 5 images are appended

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

                if positive_mel.shape[0] != syncnet_mel_step_size:  # continue if all 13 coefficients are fetched
                    continue

                anchor_window = np.concatenate(window,
                                               axis=2) / 255.  # the whole window of 5 frames is then concatenated
                anchor_window = anchor_window.transpose(2, 0, 1)

                anchor_window = torch.FloatTensor(anchor_window)
                positive_mel = torch.FloatTensor(positive_mel.T).unsqueeze(0)
                #print("Going to break from infinite loop. The label picked up is: ", label)
                break  # break the while 1 loop

            if len(query.keys()) == 0:
                query.update({
                    label: (anchor_window, positive_mel)
                })
                #print("Query was updated and now has size: ", len(query.keys()))

            else:
                support.update({
                    label: (anchor_window, positive_mel)
                })
                #print("Support was updated and now has size: ", len(support.keys()))

        #print("About to return from getitem()")
        return query, support


triplet_loss = TripletMarginLoss()

device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False


def eval_model(test_data_loader, model, device='cpu'):
    eval_steps = 100
    print('Evaluating for {} steps'.format(eval_steps))
    true_positive = 0

    print("Entering test_data_loader loop")

    step = 0

    for step in range(eval_steps):
        if step >= eval_steps:
            break

        print(f"-------------------------------------Iteration: {step}-------------------------------------")
        query, support = test_dataset[step]
        step += 1
        print("Successfully obtained a query & support set")
        model.eval()
        # Retrieve anchor window and mel from query
        q_anchor_window, q_positive_mel = query[list(query.keys())[0]][0], query[list(query.keys())[0]][1]
        q_label = list(query.keys())[0]
        cosine_loss_audio = []
        cosine_loss_frame = []
        emotion_dict = {}
        i = 0

        for label, tuple in support.items():
            emotion_dict.update({i: label})
            i = i + 1
            # Retrieve anchor window and mel from support
            s_anchor_window, s_positive_mel = tuple[0], tuple[1]

            # Reshape & send support data to GPU
            s_anchor_window = s_anchor_window.unsqueeze(0)
            s_positive_mel = s_positive_mel.unsqueeze(0)
            # s_anchor_window = s_anchor_window.to(device)
            # s_positive_mel = s_positive_mel.to(device)

            #print("Printing support shapes")
            #print(s_anchor_window.shape)
            #print(s_positive_mel.shape)

            # Reshape & send query data to GPU
            #print(q_anchor_window.dtype)

            if q_anchor_window.shape != (1, 15, 96, 96):
                q_anchor_window = q_anchor_window.unsqueeze(0)
                q_positive_mel = q_positive_mel.unsqueeze(0)
                # q_anchor_window = q_anchor_window.to(device)
                # q_positive_mel = q_positive_mel.to(device)

            #print("Printing query shapes")
            #print(q_anchor_window.shape)
            #print(q_positive_mel.shape)

            # Extract feature vectors from support anchor window and mel, and query...
            s_audio_fv, s_frame_fv = model(s_positive_mel, s_anchor_window)
            q_audio_fv, q_frame_fv = model(q_positive_mel, q_anchor_window)

            #print("FV's extracted")
            # Convert to CPU memory
            # s_audio_fv, q_audio_fv = s_audio_fv.cpu(), q_audio_fv.cpu()
            # s_frame_fv, q_frame_fv = s_frame_fv.cpu(), q_frame_fv.cpu()

            # Calculate cosine similarity between support and query for audio
            cosine_loss_audio.append(cosine_similarity(s_audio_fv, q_audio_fv))

            # Calculate cosine similarity between support and query for frames
            cosine_loss_frame.append(cosine_similarity(s_frame_fv, q_frame_fv))
            print("---PROCESSED A SUPPORT PAIR---")

        emotion_idx = np.argmax(np.mean(np.array([cosine_loss_frame, cosine_loss_audio]), axis=0))
        predicted_emotion = emotion_dict[emotion_idx]
        if predicted_emotion == q_label:
            print("Got a hit")
            true_positive += 1

    if step >= eval_steps:
        accuracy = true_positive / eval_steps

        print(f'Accuracy, is {true_positive}/{eval_steps} or {accuracy}')

        return accuracy


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
    test_dataset = DatasetTest('val')

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=2,
        num_workers=4)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncNet().to("cpu")
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)

    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    with torch.no_grad():
        eval_model(test_data_loader, model)
