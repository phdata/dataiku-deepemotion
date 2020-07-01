# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu


# Read recipe inputs
emotion_test = dataiku.Dataset("TestingImages")
emotion_test_df = emotion_test.get_dataframe()

local_images = dataiku.Folder("StYjUWGk")
local_images_info = local_images.get_info()

# Write recipe outputs
test_images = dataiku.Folder("cUcmzSP8")
test_images_info = test_images.get_info()

# Clear recipe output before writing 
test_images.clear()


for ind, row in emotion_test_df.iterrows():
    stream = local_images.get_download_stream(row.image_path)
    test_images.upload_stream(row.image_path, stream)