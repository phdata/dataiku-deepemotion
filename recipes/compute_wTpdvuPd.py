# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

from custom_video_processing import extract_frame

# Read recipe inputs
videos_to_Score = dataiku.Folder("MzP4vBYB")
videos_to_Score_info = videos_to_Score.get_info()




# Write recipe outputs
images_to_Score = dataiku.Folder("wTpdvuPd")
images_to_Score_info = images_to_Score.get_info()



df = pd.DataFrame()
df['video_path'] = videos_to_Score.list_paths_in_partition()

frames = range(0,105,5)

new_df = []

for index, item in df.iterrows():
    for f in frames:
        item['frame'] = f
        item['image_path'] = item.video_path.replace('.mp4', '_f{}.png'.format(str(f).zfill(3)))

        new_df.append(item.copy())
        
df = pd.DataFrame(new_df).reset_index(drop=True)

for index, row in df.iterrows():
    _, frame_data = extract_frame(row.video_path, videos_to_Score, 
                               row.frame)
    
    images_to_Score.upload_data(row.image_path, frame_data)
