
from  dataiku.apinode.predict.predictor import ClassificationPredictor

import dataiku
import base64

import pandas as pd
class MyPredictor(ClassificationPredictor):
    """The class for a classification Custom API node predictor"""

    def __init__(self, data_folder = None):
        """data_folder is the absolute path to the managed folder storing the data for the model
        (if any)"""
        self.data_folder = data_folder

    def predict(self, features_df):
        """
        The main prediction method.

        :param: df: a dataframe of 1 or several records to predict

        :return: Either:
            ``decision_series`` or
            ``(decision_series, proba_df)`` or
            ``(decision_series, proba_df, custom_keys_list)``

        decision_series must be a Pandas Series of decisions

        proba_df is optional and must contain one column per class

        custom_keys_list is optional and must contain one entry per input row. Each entry of
        custom_keys_list must be a Python dictionary. These custom keys will be sent in the
        output result

        decision_series, proba_df and custom_keys_list must have the same number of rows than df.
        """
        
        dataiku.set_remote_dss('http://52.71.116.91:80', 'hTeQ6RbM8u4ASGMthBChKxDfniusNoZu')

        df = features_df
        data_folder = dataiku.Folder('MzP4vBYB', project_key='PHDATAEMOTION')
        
        for ind, row in df.iterrows():
            fname = 'ind_{}.mp4'.format(ind)

            df.loc[ind, 'fname'] = fname
            data_folder.upload_data(fname, base64.urlsafe_b64decode(row.b64_video.encode('utf-8')))

        client = dataiku.api_client()
        project = client.get_project('PHDATAEMOTION')
        scenario = project.get_scenario('PRODSCOREVIDEOS')
        
        scenario_run = scenario.run_and_wait()
        success = scenario_run.get_info()['result']['outcome'] == 'SUCCESS'

        ds = project.get_dataset('ProdScoredVideos')
        scores = pd.DataFrame(data=list(ds.iter_rows()), 
                              columns=[c['name'] for c in ds.get_schema()['columns']])
        scores = scores.set_index('video_path')
    
    
        emotions = ['calm', 'sad', 'surprised', 'neutral', 
            'fearful', 'angry', 'happy', 'disgust']

        df = df.drop(columns=['b64_video']).join(scores, on='fname')

        for ind, row in df.iterrows():
            max_val = 0
            max_label = None
            sum_val = 0

            for e in emotions:
                p = row['prediction_{}_avg'.format(e)]
                sum_val += p
                if p > max_val:
                    max_val = p
                    max_label = e

            df.loc[ind, 'prediction'] = max_label

            for e in emotions:
                df.loc[ind, 'proba_{}'.format(e)] = row['prediction_{}_avg'.format(e)] / sum_val
                
        decisions = df.prediction
        proba_df = df[['proba_{}'.format(e) for e in emotions]]
            
        return (decisions, proba_df)