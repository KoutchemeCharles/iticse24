from copy import deepcopy
from dotmap import DotMap
from sklearn.model_selection import ParameterGrid
from src.utils.utils import format_string
from numpy import argmax


def run(self):
    df = self.dataset_handler.dataset
    if self.test_run: df = df.iloc[:8]

    suffix = ""
    annot_cols = ["prompt_en", "annotator", 
                  "issues_by_annotator", "grading_value"]
    df = df.drop(columns=annot_cols)
    
    if self.config.hyperparameters.name == "search":
        grids = ParameterGrid(self.config.hyperparameters.grids)
        if self.test_run: grids = list(grids)[:1]
        gen_param = self.search_best_gen_params(grids)
        suffix = "best"
    elif self.config.hyperparameters.name == "default":
         gen_param = self.config.hyperparameters.params.toDict()
         suffix = "default"
    else:
        raise ValueError("Unknown hyperparameter search strategy")
        
    results = self._generate_and_evaluate(df, gen_param)
    self._save_results(results, gen_param, self.results_save_dir, suffix)

    return results 

def _generate(self, df, gen_param):
    task = self.config.task.toDict()
    # We only do zero-shot prompting for LLMs 
    if task["type"] == "zero_shot":
        grading_df = self._zero_shot_prompt(df, gen_param)
    else:
        raise ValueError("Type of prompting not implemented yet")

    return grading_df


def _evaluate(self, grading_df, gen_param):
    grading_df = self.grade_feedback_with_judge(grading_df)
    return {
        "eval_ds": grading_df.to_dict(),
        "scores": self.compute_scores(grading_df),
        "hyperparameters": gen_param,
    } 


def _generate_and_evaluate(self, df, gen_param):
    return self._evaluate(self._generate(df, gen_param), gen_param)


def _zero_shot_prompt(self, df, gen_param):
    return (df.groupby("id", as_index=False, group_keys=False)
              .apply(self.create_zero_shot_messages, 
                     gen_param=gen_param))


def create_zero_shot_messages(self, group, gen_param):
    instructions = self.config.task.toDict()["instructions"]
    # format the instructions according to the criteria
    instr = deepcopy(instructions)
    instr[1]["content"] = build_query(group, instr[1]["content"])
    # get the model answer 
    group = self.query_and_broadcast(instr, group, gen_param)
    
    return group 


def query_and_broadcast(self, instr, group, gen_param):
    # Makes sure we have the right format 
    instr = clean_messages(instr)
    # gathering all information from the judge grading
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in instr])
    outputs = self.agent.query([instr], **gen_param)  # still using instr
    full_chat = outputs[0] # assuming a single response generated

    group["feedback_model"] = self.agent.name 
    group["prompt_en"] = prompt
    group["full_chat"] = full_chat
    group["feedback"] = extract_feedback(full_chat)

    return group 

def search_best_gen_params(self, gen_param_space):
    """ 
    Searches for the generation parameters which will
    produce the best results on the dev set. 
    """

    df = self.dataset_handler.get_split("val")
    annot_cols = ["prompt_en", "annotator", 
                  "issues_by_annotator", "grading_value"]
    df = df.drop(columns=annot_cols)

    if self.test_run: gen_param_space = list(gen_param_space)[:1]
    
    scores = []
    for gen_param in gen_param_space:
        print("Evaluating hyperapameters", gen_param)
        results = self._generate_and_evaluate(df, gen_param)
        scores.append(self.compute_objective(results["scores"]))

    best_gen_param = gen_param_space[argmax(scores)]

    print("Found best generation parameter for config", self.config)
    print(best_gen_param)

    return best_gen_param


def compute_objective(self, scores):
    """
    Compute a single score for selecting the right hyperparameters
    for evaluation. Currently, we choose the score of the combination
    of all grading criteria.
    """
    
    max_combi = max([s["n_combi"] for s in scores])
    return sum([s["score_value"] for s in scores if max_combi])


def build_query(group, unformated_string):
    # many of the columns contain the information 
    # which we extract for formatting part of the instructions
    # given the templates
    # we need a copy here 
    info_source = group.iloc[0].to_dict()
    info_source["criteria"] = "\n".join(f"({i}): {c}" for i, c in 
                                        enumerate(group["grading_criteria"]))
    return format_string(DotMap(info_source), unformated_string)


def extract_feedback(full_chat):
    beacon = "## Feedback:" # TODO: should be more dynamic how it's found
    # if not beacon in full_chat: 
        #warn("Beacon feedback not found in {full}")
    
    # ChatGPT does not output the original prompt 
    start_index = 0
    if beacon in full_chat:
        # Open-source models generate that 
        start_index = full_chat.find(beacon) + len(beacon)
    
    feedback = full_chat[start_index:] 
    lines = feedback.splitlines()
    lines = [l for l in lines if not l.startswith("<|")]
    return "\n".join(lines)


def clean_messages(messages):
    """ 
    Merges messages of the same role together if they follow each other.

    """
    new_messages = [messages[0]]
    for message in messages[1:]:
        if message["role"] == new_messages[-1]["role"]:
            new_messages[-1]["content"] += ("\n" + message['content'])
        else:
            new_messages.append(message)

    return new_messages