"""
Helper functions to calculate metrics like KS, Gini, Precision, Recall, F1 Score,
Population Stability Index, Charecteristic Stability Index, Confusion Matrix,
Chi Square with IV (Informational Value)
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, precision_recall_curve

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)-8s | %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
_logger = logging.getLogger("ML UTILS")

class Metrics:
    def __init__(self, actuals, predictions):
        self._actuals = actuals
        self._predictions = predictions
        self._df = pd.DataFrame({'actuals': actuals, 'predictions': predictions})
        self.auc_threshold = self._get_auc_threshold
        self.precision_recall_threshold = self._get_precision_recall_threshold
        self.tn, self.fp, self.fn, self.tp, self.precision, self.recall, self.f1_score = self.precision_recall_f1_score()

    @property
    def _get_auc_threshold(self):
        fpr,tpr,threshold = roc_curve(self._actuals,self._predictions) 
        gmean = np.sqrt(tpr*(1-fpr))
        youdenJ = tpr-fpr
        threshold_gmean = round(threshold[np.argmax(gmean)], 4)
        threshold_yJ = round(threshold[np.argmax(youdenJ)], 4)
        threshold_cutoff = max(threshold_gmean, threshold_yJ)
        return float(threshold_cutoff)
    
    @property
    def _get_precision_recall_threshold(self):
        precision, recall, thresholds = precision_recall_curve(self._actuals, self._predictions)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        max_f1_index = np.argmax(f1_scores)
        return float(thresholds[max_f1_index])

    @property
    def gini(self):
        """
        Returns Gini value calculated from actual and predicted using
        `sklearn.metrics.roc_curve` and `sklearn.metrics.auc`.
        """
        fpr, tpr, _ = roc_curve(self._actuals, self._predictions)
        auroc = auc(fpr, tpr)
        gini = 2*auroc - 1
        return float(gini)

    def precision_recall_f1_score(self):
        """
        Calculates TN, FP, FN, TP, Precision, Recall, F1 Score using Optimal Threshold value. 
        """
        threshold_cutoff = self._get_precision_recall_threshold
        self.y_pred=np.where(self._predictions>=threshold_cutoff,1,0)
        tn, fp, fn, tp = confusion_matrix(self._actuals, self.y_pred).ravel()
        precision = precision_score(self._actuals, self.y_pred)
        recall = recall_score(self._actuals, self.y_pred)
        f1 = f1_score(self._actuals, self.y_pred)
        return float(tn), float(fp), float(fn), float(tp), float(precision), float(recall), float(f1)
    
    def to_dict(self):
        """Returns all calculated metrics in a `dict` form."""
        return {
            'gini': self.gini,
            'threshold': self.auc_threshold,
            'tn': self.tn, 'fp': self.fp, 'fn': self.fn, 'tp': self.tp,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score
            }

    @property
    def values(self):
        """
        Returns metrics like KS, Gini, Optimal Threshold, TN, FP, FN, TP, Precision, Recall and F1 Score
        from `ml_utils.measure.metrics.Metrics` object.
        """
        return self.to_dict()
    
    def decilewise_counts(self, bins=10, cutpoints=None):
        """
        Returns a summarized pandas dataframe with total and responders for each decile based on `positive_probability`.
        """
        if cutpoints is None:
            cutpoints = self._df['predictions'].quantile(np.arange(0,bins+1)/bins).reset_index(drop=True)
            cutpoints = sorted(list(set([0]+list(cutpoints)[1:-1]+[1])))
        self._df['bins'] = pd.cut(self._df['predictions'], cutpoints, duplicates='drop')
        out_df = self._df.groupby('bins')['actuals'].agg(['count','sum']).sort_values(by=['bins'], ascending=False).reset_index()
        out_df.columns = ['band','total','resp']
        return out_df