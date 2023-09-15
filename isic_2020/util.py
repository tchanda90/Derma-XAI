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
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return roc_t['threshold'].item()


def get_char_predictions(trainer, model, dataloader, threshold=0.5):
    """
    Stores predictions, scores, and true values in a DataFrame.
    """

    predictions = trainer.predict(model, dataloader)

    dfs = []

    for preds in predictions:
        y_pred, y_true, image_name = preds
        df_true = pd.DataFrame(y_true, columns=char_class_labels)
        df_score = pd.DataFrame(y_pred, columns=char_class_labels_score)

        y_pred = torch.where(y_pred >= threshold, 1, 0)
        df_pred = pd.DataFrame(y_pred, columns=char_class_labels_pred)

        df_name = pd.DataFrame(image_name, columns=['image'])

        df = pd.concat([df_name, df_true, df_pred, df_score], axis=1)

        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)

    result = pd.concat([model.test_set.metadata[["image_id", "lesion_id", "benign_malignant"]].reset_index(drop=True),
                        result.reset_index(drop=True)], axis=1)

    return result


def get_dx_predictions(trainer, model, split='test', threshold=0.5):
    """
    Stores predictions, scores, and true values in a DataFrame.
    """
    if split == 'test':
        predictions = trainer.predict(model, model.test_dataloader())
    else:
        predictions = trainer.predict(model, model.val_dataloader())

    dfs = []

    for preds in predictions:

        y_pred, y_true, image_name = preds
        df = pd.DataFrame(y_pred, columns=['score'])
        y_pred = torch.where(y_pred >= threshold, 1, 0)
        
        df['pred'] = y_pred
        df['true'] = y_true

        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True).dropna()

    return result


def display_scores(result):
    true = 'true'
    score = 'score'
    pred = 'pred'
    print('=====')
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