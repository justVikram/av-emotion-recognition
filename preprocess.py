import cv2 as cv
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt


def preprocess_vid(source_path, output_path):
    count = 0
    for vid in os.listdir(source_path):
        if vid.endswith('.mp4'):
            count = count + 1
            vid = os.path.join(source_path, vid)
            split_video(vid, output_path, count)


# Function to split video into chunks of duration = 2 seconds and save them in a folder
def split_video(video_path, output_path, count):
    # Code to get duration of input video
    cap = cv.VideoCapture(video_path)
    duration = int(cap.get(cv.CAP_PROP_FRAME_COUNT) / cap.get(cv.CAP_PROP_FPS))
    cap.release()

    new_output_path = os.path.join(output_path, f'video_{str(count)}')
    os.mkdir(new_output_path)
    os.mkdir(os.path.join(new_output_path, 'segments'))

    seek = 0
    while duration - seek >= 2:
        start = seek
        end = seek + 2
        os.system(
            f'ffmpeg -i {video_path} -ss 00:00:0{start} -t 00:00:02 -c:v libx264 -c:a copy {new_output_path}/segments/segment{start}-{end}.mp4')
        seek = seek + 2

    # Get frames for each segment
    frames_path = os.path.join(new_output_path, 'frames')
    os.mkdir(frames_path)
    audio_path = os.path.join(new_output_path, 'audio')
    os.mkdir(audio_path)
    mfcc_path = os.path.join(new_output_path, 'mfcc')
    os.mkdir(mfcc_path)

    segment_count = 0
    for segment in os.listdir(os.path.join(new_output_path, 'segments')):
        if segment.endswith('.mp4'):
            segment_count = segment_count + 1
            segment_to_frame_dir = os.path.join(frames_path, f'frames_for_segment_{segment_count}')
            os.mkdir(segment_to_frame_dir)
            segment_to_audio_dir = os.path.join(audio_path, f'audio_for_segment_{segment_count}')
            os.mkdir(segment_to_audio_dir)
            mfcc_to_segment_dir = os.path.join(mfcc_path, f'mfcc_for_segment_{segment_count}')
            os.mkdir(mfcc_to_segment_dir)
            get_frames(os.path.join(f'{new_output_path}/segments', segment), segment_to_frame_dir)
            extract_audio(os.path.join(f'{new_output_path}/segments', segment), segment_to_audio_dir)
            extract_mfcc(os.path.join(segment_to_audio_dir, 'output-audio.aac'), mfcc_to_segment_dir)


# Function to extract audio from segment
def extract_audio(video_path, output_path):
    os.system(f'ffmpeg -i {video_path} -vn -acodec copy {output_path}/output-audio.aac')


def extract_mfcc(audio_path, output_path):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.savefig(os.path.join(output_path, 'mfcc.png'))


# Function to get frames from video into frames
def get_frames(video_path, output_path):
    cap = cv.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            frame_count = frame_count + 1
            cv.imwrite(os.path.join(os.path.join(output_path), f'frame_{frame_count}.jpg'), frame)
        else:
            break

    cap.release()


if __name__ == '__main__':
    source_path = '/Users/avikram/Projects/av-emotion-recognition/data/input'
    output_path = '/Users/avikram/Projects/av-emotion-recognition/data/output'
    preprocess_vid(source_path, output_path)
