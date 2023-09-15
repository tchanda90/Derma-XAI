import pandas as pd
from sklearn import metrics
from config import *


def find_optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    threshold = []
    for col in char_class_labels:
        fpr, tpr, threshold = metrics.roc_curve(target[col], predicted[col+'_score'])
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

        threshold.append(roc_t['threshold'].item())
    return threshold


def get_char_predictions(trainer, model, threshold=0.5, split='test'):
    """
    Stores predictions, scores, and true values in a DataFrame.
    """
    
    if split == 'test':
        predictions = trainer.predict(model, model.test_dataloader())
        metadata_df = model.test_set.metadata[["image_id", "lesion_id", "benign_malignant"]]
    elif split == 'val':
        predictions = trainer.predict(model, model.val_dataloader())
        metadata_df = model.val_set.metadata[["image_id", "lesion_id", "benign_malignant"]]

    dfs = []

    for preds in predictions:
        y_pred, attributions, y_true, image_id, x = preds
        
        df_img = pd.DataFrame(image_id, columns=['image_id'])
        
        df_true = pd.DataFrame(y_true, columns=char_class_labels)
        df_score = pd.DataFrame(y_pred, columns=char_class_labels_score)

        y_pred = torch.where(y_pred >= threshold, 1, 0)
        df_pred = pd.DataFrame(y_pred, columns=char_class_labels_pred)

        df = pd.concat([df_img, df_true, df_pred, df_score], axis=1)
        

        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

    result = pd.merge(metadata_df, result, on='image_id')

    return result


def get_dx_predictions(trainer, model):
    """
    Stores predictions, scores, and true values in a DataFrame.
    """

    predictions = trainer.predict(model)

    dfs = []

    for preds in predictions:

        y_pred, y_true = preds

        df = pd.DataFrame(y_pred, columns=['score'])
        df['pred'] = df['score'].round()
        df['true'] = pd.DataFrame(y_true, columns=['true'])

        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True).dropna()

    return result


def display_scores(result):

    for true, pred, score in zip(char_class_labels, char_class_labels_pred, char_class_labels_score):
        print('\n=====')
        print(true)
        print('AUC:', metrics.roc_auc_score(result[true], result[score]))
        print('Balanced Acc:', metrics.balanced_accuracy_score(result[true], result[pred]))
        print('Sensitivity:', metrics.recall_score(result[true], result[pred]))
        print('Specificity:', metrics.recall_score(result[true], result[pred], pos_label=0))
        print('=====\n')



def confidence_interval(data, size=10000, func=np.mean):
    """creates a bootstrap sample, computes replicates and returns replicates array"""
    # Create an empty array to store replicates
    bs_replicates = np.empty(size)
    
    np.random.seed(42)
    # Create bootstrap replicates as much as size
    for i in range(size):
        # Create a bootstrap sample
        bs_sample = np.random.choice(data,size=len(data))
        # Get bootstrap replicate and append to bs_replicates
        bs_replicates[i] = func(bs_sample)
    
    return np.percentile(bs_replicates, [2.5, 97.5])
        
        
        
import random
import numpy as np

from torch.utils.data.sampler import Sampler


class MultilabelBalancedRandomSampler(Sampler):
    """
    MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
    number of classes n_classes, samples from the data with equal probability per class
    effectively oversampling minority classes and undersampling majority classes at the
    same time. Note that using this sampler does not guarantee that the distribution of
    classes in the output samples will be uniform, since the dataset is multilabel and
    sampling is based on a single class. This does however guarantee that all classes
    will have at least batch_size / n_classes samples as batch_size approaches infinity
    """

    def __init__(self, labels, indices=None, class_choice="least_sampled"):
        """
        Parameters:
        -----------
            labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)
            indices: an arbitrary-length 1-dimensional numpy array representing a list
            of indices to sample only from
            class_choice: a string indicating how class will be selected for every
            sample:
                "least_sampled": class with the least number of sampled labels so far
                "random": class is chosen uniformly at random
                "cycle": the sampler cycles through the classes sequentially
        """
        self.labels = labels
        self.indices = indices
        if self.indices is None:
            self.indices = range(len(labels))

        self.num_classes = self.labels.shape[1]

        # List of lists of example indices per class
        self.class_indices = []
        for class_ in range(self.num_classes):
            lst = np.where(self.labels[:, class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.class_indices.append(lst)

        self.counts = [0] * self.num_classes

        assert class_choice in ["least_sampled", "random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.indices):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        class_ = self.get_class()
        class_indices = self.class_indices[class_]
        chosen_index = np.random.choice(class_indices)
        if self.class_choice == "least_sampled":
            for class_, indicator in enumerate(self.labels[chosen_index]):
                if indicator == 1:
                    self.counts[class_] += 1
        return chosen_index

    def get_class(self):
        if self.class_choice == "random":
            class_ = random.randint(0, self.labels.shape[1] - 1)
        elif self.class_choice == "cycle":
            class_ = self.current_class
            self.current_class = (self.current_class + 1) % self.labels.shape[1]
        elif self.class_choice == "least_sampled":
            min_count = self.counts[0]
            min_classes = [0]
            for class_ in range(1, self.num_classes):
                if self.counts[class_] < min_count:
                    min_count = self.counts[class_]
                    min_classes = [class_]
                if self.counts[class_] == min_count:
                    min_classes.append(class_)
            class_ = np.random.choice(min_classes)
        return class_

    def __len__(self):
        return len(self.indices)
    
    