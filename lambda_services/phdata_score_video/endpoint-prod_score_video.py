
from  dataiku.apinode.predict.predictor import ClassificationPredictor
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

        # Note: this sample uses the second form (decision_series, proba_df)

        # Note: this sample "cheats" and always returns 5 predictions.
        # You should actually return 1 prediction per row in the features_df

        print "Features DataFrame %s" % features_df

        # predictions, one per record (features_df row)
        predictions = pd.Series(["good", "fair", "poor", "good", "poor"])

        # optional probas for each class (may be None or a DataFrame with one column per class)
        probas = pd.DataFrame({
            'proba_good': pd.Series([.9, .6, .2, .7, .2]),
            'proba_fair': pd.Series([.2, .7, .3, .3, .3]),
            'proba_poor': pd.Series([.2, .6, .6, .3, .9])
        })

        return (predictions, probas)
