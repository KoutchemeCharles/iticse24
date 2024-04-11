from sklearn.metrics import (
    accuracy_score,
    precision_score,
    fbeta_score,
    cohen_kappa_score,
    recall_score
)

def compute_scores(self, df):
    """ 
    Evaluate the performance of the judge generation given
    the ground truth annotations.

    Parameters
    ----------
    df: pandas DataFrame
        Contains the judge evaluation as well as the ground truth

    Results
    -------
    scores: dict
        Performance of the judge generations

    """

    requested_scorings = self.config.task.scoring
    results = []
    for scoring in requested_scorings:
        if scoring["metric_name"] == "accuracy":
            scoring_f = lambda df: compute_accuracy(df)
        elif scoring["metric_name"] == "precision":
            scoring_f = lambda df: compute_precision(df)
        elif scoring["metric_name"] == "recall":
            scoring_f = lambda df: compute_recall(df)
        elif scoring["metric_name"] == "f1_score":
            scoring_f = lambda df: compute_fbeta(df, beta=1)
        elif scoring["metric_name"] == "f0.5_score":
            scoring_f = lambda df: compute_fbeta(df, beta=0.5)
        elif scoring["metric_name"] == "kappa":
            scoring_f = lambda df: compute_kappa(df)
        else:
            raise ValueError(f"Unkwown scoring {scoring['metric_name']}")
        
        
        if scoring["grouping"] != "all":
            values = df.groupby(scoring["grouping"]).apply(scoring_f)
            scoring.update(values.to_dict())    
            results.append(scoring)

    # results.append(df.groupby("id").apply(all_scoring).mean())
        
    return results


def all_scoring(sub_df):
    return (sub_df.grading_value == sub_df.grading_value).all()

def compute_kappa(df):
    return cohen_kappa_score(y1=df.grading_value, y2=df.judge_answer)

def compute_fbeta(df, beta):
    return fbeta_score(y_true=df.grading_value, 
                       y_pred=df.judge_answer,
                       beta=beta, 
                       zero_division=1.0) # if there is no predicted sample

def compute_accuracy(df):
    return accuracy_score(y_true=df.grading_value, 
                          y_pred=df.judge_answer)
    

def compute_precision(df):
    return precision_score(y_true=df.grading_value, y_pred=df.judge_answer)


def compute_recall(df):
    return recall_score(y_true=df.grading_value, y_pred=df.judge_answer)
