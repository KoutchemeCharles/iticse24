import pandas as pd
import numpy as np 
from src.grading.Grading import Grading
from src.utils.files import read_config
from itertools import combinations
from dotmap import DotMap 

def grade_feedback_with_judge(self, df):
    """ 
    Use a Judge LLM to evaluate the quality of the generated
    feedback. 
    """

    grader_config = DotMap({
        "hyperparameters": read_config(self.config.task.scoring.hyperparameters),
        "agent": read_config(self.config.task.scoring.model),
        "task": read_config(self.config.task.scoring.task),
    })

    self.grader = Grading.only_for_grading(grader_config)
    gen_param = grader_config.hyperparameters.params
    grading_df = self.grader._generate(df, gen_param)

    return grading_df


def compute_scores(self, df):
    """ # metric_name, grouping
    all_scores = []
    all_scores.append({
        "metric_name": "accuracy", 
        "grouping": "all", 
        "score_value": float(df.judge_answer.mean())
    })
    scores = df.groupby("grading_criteria").judge_answer.mean().to_dict()
    all_scores.append({
        "metric_name": "accuracy", 
        "grouping": "grading_criteria",
        # avoid json serialization errors
        **{key: float(value) for key, value in scores.items()}
    })

    # take all combinations of criteria and count pairs where they are true 

    grouped = df.groupby("id").apply(count_pairs)
    scores = grouped.groupby("combi").mean()
    all_scores.append({
        "metric_name": "accuracy", 
        "grouping": "combinations",
        # avoid json serialization errors
        **{key: float(value) for key, value in scores.items()}
    })
    """
    
    grouped = df.groupby("id").apply(count_pairs)
    return grouped.groupby("combi").mean().reset_index().to_dict("records")


def count_pairs(group):
    combis = []
    for i in range(1, len(group["grading_criteria"]) + 1):
        combis.extend(combinations(group["grading_criteria"], i))
    
    infos = []
    for combi in combis:
        # careful here with interpretation, we are looking for
        # groups where that specific combi is exactly respected

        sub_df = group.set_index("grading_criteria")
        mask = np.array([True if gc in combi else False for gc in sub_df.index])
        score = (sub_df.judge_answer.values == mask).all()

        infos.append({
            "n_combi": len(combi),
            "combi": "_".join(combi),
            "score_value": score
        })

    return pd.DataFrame(infos)