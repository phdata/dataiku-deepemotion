import os
from tempfile import mkstemp

import cv2
import numpy as np

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
