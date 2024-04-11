"""
Wrapper class for accessing the ICER data. 
"""

import pandas as pd 
from markdownify import markdownify as md

class ICER():

    def __init__(self, config) -> None:
        self.config = config 
        self.__dataset = self.__load_dataset()

    def __load_dataset(self):
        df = pd.read_csv(self.config.path)
        df = df.reset_index(drop=False).rename(columns={"index": "id"})
        df = df.iloc[:, :-4]
        df = pd.melt(df, 
                id_vars=df.columns[:-4], 
                var_name="grading_criteria", 
                value_name="grading_value")
        df = df.sort_values(by=["id", "grading_criteria"])
        df["annotator"] = ""
        df["feedback_model"] = "gpt-3.5"
        df["feedback"] = df["gpt35_en"].apply(lambda s: s.strip())
        df["handout"] = df.prompt_en.apply(extract_english_handout).apply(md)
        df["grading_criteria"] = [c[:-1] if c.endswith(".") else c for c in df["grading_criteria"]]
        df = df[["Produces" not in c for c in df.grading_criteria]]
        df["grading_value"] = df["grading_value"].astype(bool)

        # Change two of the criteria such that "a positive value" means better feedback
        repurpose = {
            "Identifies non-existent issues": "Does not identify non-existent issues",
            "Produces unnecessary, duplicate, or repetitive content": "Does not produce unnecessary, duplicate, or repetitive content",
        }
                
        mask = df.grading_criteria.isin(repurpose.keys())
        df.loc[mask, "grading_value"] = ~df.loc[mask, "grading_value"]
        df = df.replace(repurpose)

        # change the code a bit 
        df["submitted_code"] = [f"```dart\n{c.strip()}\n```" for c in df["submitted_code"]]
        df["sample_solution"] = [f"```dart\n{c.strip()}\n```" for c in df["sample_solution"]]
        df = df.drop(columns=["gpt35_en", "assignment_handout"])
        df = df.rename(columns={"issues_by_arto": "issues_by_annotator"})
        
        return df 
    
    @property
    def dataset(self):
        return self.__dataset.reset_index(drop=True) 
    
    def get_split(self, split):
        df = self.dataset 
        if split == "val":
            groups = df.groupby("handout", as_index=False, group_keys=False)
            df =  groups.apply(take_n_groups)
            return df
        else:
            return df 


def take_n_groups(df, n=1):
    groups = df.groupby("id", as_index=False, group_keys=False)
    groups = list(groups.groups.values())[:1]
    df = pd.concat([df.loc[ix] for ix in groups])
    return df 

def extract_english_handout(prompt_en):
    """ 
    Extract the assignment handout in english from the prompt
    used to obtain the feedback from the model
    """
    start_beacon = "## Programming exercise handout\n"
    end_beacon = "## My code\n"
    start_index = prompt_en.index(start_beacon) + len(start_beacon)
    end_index = prompt_en.index(end_beacon)
    return prompt_en[start_index: end_index].strip()