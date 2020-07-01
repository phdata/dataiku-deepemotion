# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

import os
from tempfile import mkstemp

import cv2


# Code to extract frame from a video
def extract_frame(video_path, video_folder, frame_pct):
    try:
        vf, vfname = mkstemp(suffix='.mp4')
        vfile = os.fdopen(vf, 'wb')

        p = video_path

        # We use get_download_stream to be compatible with S3-based folders
        with video_folder.get_download_stream(p) as stream:
            vfile.write(stream.read())
        vfile.close()

        vid_obj = cv2.VideoCapture(vfname)

        if not vid_obj.isOpened():
            return None

        frame_count = vid_obj.get(cv2.CAP_PROP_FRAME_COUNT)
        fp = frame_pct

        frame_number = np.max([
                int(np.round(frame_count * (fp / 100.0))) - 2,
                0
            ])

        vid_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, image = vid_obj.read()

        if not success:
            return None

        success, buffer = cv2.imencode('.png', image)

        if not success:
            return None

    finally:
        os.remove(vfname)

    return image, buffer.tobytes()

# Decode emotion
def get_emotion(fname):
    code = fname.split('-')[2]

    if code == '01':
        return 'neutral'
    elif code == '02':
        return 'calm'
    elif code == '03':
        return 'happy'
    elif code == '04':
        return 'sad'
    elif code == '05':
        return 'angry'
    elif code == '06':
        return 'fearful'
    elif code == '07':
        return 'disgust'
    elif code == '08':
        return 'surprised'
    
# Decode intensity
def get_intensity(fname):
    code = fname.split('-')[3]

    return 'normal' if code == '01' else 'strong'


# Read recipe inputs -- CHANGE THIS IDENTIFIER TO MATCH YOUR INPUT FOLDER
emotion_videos = dataiku.Folder("JV4cw0cu")
emotion_videos_info = emotion_videos.get_info()

# Write recipe output files -- CHANGE THIS IDENTIFIER TO MATCH YOUR OUTPUT FOLDER
emotion_images = dataiku.Folder("StYjUWGk")
emotion_images_info = emotion_images.get_info()

# Read video file names and build dataframe
df = pd.DataFrame()
df['video_path'] = emotion_videos.list_paths_in_partition()

df['emotion'] = df.video_path.apply(get_emotion)
df['intensity'] = df.video_path.apply(get_intensity)
df['statement'] = df.video_path.apply(lambda s: s.split('-')[4])
df['repetition'] = df.video_path.apply(lambda s: s.split('-')[5])
df['actor'] = df.video_path.apply(lambda s: s.split('-')[6].split('.')[0])

# Expand df to include rows for each image we want
# Write out frame percentage -- e.g. 0 means first frame, 100 is last frame, etc.
frames = range(0,105,5)

new_df = []

for index, item in df.iterrows():
    for f in frames:
        item['frame'] = f
        item['image_path'] = item.video_path[1:].replace('.mp4', '_{}_f{}.png'.format(item.emotion, f))

        new_df.append(item.copy())

df = pd.DataFrame(new_df).reset_index(drop=True)


# Extract all frames
for index, row in df.iterrows():
    # Skip images we've done before...
    path_details = emotion_images.get_path_details(row.image_path)
    if path_details['exists']:
        continue

    _, frame_data = extract_frame(row.video_path, emotion_videos, 
                               row.frame)

    emotion_images.upload_data(row.image_path, frame_data)

# Write output CSV
emotion_images_csv = dataiku.Dataset("EmotionImagesCSV")
emotion_images_csv.write_with_schema(df)