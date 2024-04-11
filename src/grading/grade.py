import re
import numpy as np 
import pandas as pd 

from warnings import warn
from random import shuffle
from dotmap import DotMap
from copy import deepcopy
from itertools import islice
from sklearn.model_selection import ParameterGrid

from src.utils.utils import format_string

def run(self):
    df = self.dataset_handler.dataset
    if self.test_run: df = df.iloc[:8]
    
    if self.config.hyperparameters.name == "search":
        grids = ParameterGrid(self.config.hyperparameters.grids)
        if self.test_run: grids = list(grids)[:1]
        gen_param = self.search_best_gen_params(grids)
    elif self.config.hyperparameters.name == "default":
         gen_param = self.config.hyperparameters.params.toDict()
    else:
        raise ValueError("Unknown hyperparameter search strategy")
        
    results = self._generate_and_evaluate(df, gen_param)
    self._save_results(results, gen_param, self.results_save_dir, "")

    return results 


def _generate(self, df, gen_param):
    task = self.config.task.toDict()
    if task["type"] == "zero_shot":
        grading_df = self._zero_shot_prompt(df, gen_param)
    elif task["type"] == "few_shot":
        grading_df = self._few_shot_prompt(df, gen_param) 
    else:
        raise ValueError("Type of prompting not implemented yet")
    
    return grading_df


def _evaluate(self, grading_df, gen_param):
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


def _few_shot_prompt(self, df, gen_param):
    groups = df.groupby("id", as_index=False, group_keys=False).groups
    task = self.config.task.toDict()

    # Reminder: it's important not to modify the group objects when
    # handling them since we are playing with views

    new_groups = []
    for id_, index in groups.items():
        # the help request being queried for answer 
        group = df.loc[index]
        # Assemble all examples which can be used for "shoting"
        all_shots = df[(df.id != id_)]  #
        if task["strategy"]["from_same_problem"]:
            mask = all_shots.assignment_title == group.assignment_title.iloc[0]
        else:
            mask = all_shots.assignment_title != group.assignment_title.iloc[0]

        all_shots = all_shots[mask]
        if not len(all_shots):
            raise ValueError("No examples available for few shot evaluation. Could be due to test run sampling.")
        
        n_shots = task["n_shots"]
        indexes = list(all_shots.groupby("id").groups.values())
        shuffle(indexes)
        shot_indexes = islice(indexes, n_shots)
        shot_groups = [all_shots.loc[sindex] for sindex in shot_indexes]
        new_groups.append(self.create_few_shot_messages(group, shot_groups, gen_param))

    return pd.concat(new_groups, axis=0)
        

def create_zero_shot_messages(self, group, gen_param):
    instructions = self.config.task.toDict()["instructions"]
    # format the instructions according to the criteria
    instr = deepcopy(instructions)
    instr[1]["content"] = build_query(group, instr[1]["content"])
    # get the model answer 
    group = self.query_and_broadcast(instr, group, gen_param)
    
    return group 


def create_few_shot_messages(self, group, shot_groups, gen_param):
    task = self.config.task.toDict()
    repeat = task["strategy"]["repeat_task_description"]
    instr = deepcopy(self.config.task.toDict()["instructions"])
    # Let's create "messages" that are going to be sent to the
    # judge LLM using the template provided in the configuration file
    new_instructions = []
    # the first element in the instruction is the system prompt (behaviour)
    if instr[0]["content"]: new_instructions.append(instr[0])
    # the second element in the instruction is the task description
    if instr[1]["content"]: new_instructions.append(instr[1])
    # Note: the first and second element could have been empty dict
    # Now, we format the few-shot examples 
    for shot_group in shot_groups:
        # the third element represents information given for answerring
        assert instr[2]["role"] == "user"
        new_instructions.append({
            "role": "user", 
            "content": build_query(shot_group, instr[2]["content"])
        })
        # the fourth shows the expected format of an answer 
        # which we format as few_shots
        new_instructions.append({
            "role": "assistant", 
            "content": build_answer(shot_group, instr[3]['content'])
        })

        if repeat and instr[1]["content"]: new_instructions.append(instr[1])

    # finally, we query for the particular group of interest
    new_instructions.append({
        "role": "user", 
        "content": build_query(group, instr[2]['content'])
    }) 

    return self.query_and_broadcast(new_instructions, group, gen_param)


def query_and_broadcast(self, instr, group, gen_param):
    # Makes sure the format is appropriate 
    instr = clean_messages(instr)
    # gathering all information from the judge grading
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in instr])
    print("Grading: instructions sent to the judge model", instr)
    outputs = self.agent.query_with_message(instr, **gen_param)  # still using instr
    response = outputs[0] # assuming a single response generated
    print("Grading: responses generated by the judge", outputs)
    beacons = [f"({i})" for i in range(len(group))]
    answers = match_criteria(beacons, response)

    # if no answer are provided, the answer is automatically wrong
    if None in answers and "grading_value" in group:
        warn(f"For some grading we got no answer {answers}")
        answers = [a if a != None else not o 
                   for a, o in zip(answers, group["grading_value"])]
    elif None in answers and not "grading_value" in group:
        raise ValueError("No answers generated by judge in non-evaluation mode")
    
    # need to broadcast that back again to the df
    group["judge_answer"] = answers 
    group["judge_prompt"] = prompt
    group["judge_full_response"] = response

    return group 


def search_best_gen_params(self, gen_param_space):
    """ 
    Searches for the generation parameters which will
    produce the best results on the dev set. 
    """

    df = self.dataset_handler.get_split("val")
    scores = []
    for gen_param in gen_param_space:
        print("Evaluating hyperapameters", gen_param)
        results = self._generate_and_evaluate(df, gen_param)
        scores.append(self.compute_objective(results["scores"]))

    best_gen_param = gen_param_space[np.argmax(scores)]

    print("Found best generation parameter for config", self.config)
    print(best_gen_param)

    return best_gen_param


def compute_objective(self, scores):
    """ Computes a value that will be used for validation. """

    # by default, the first scoring metric is used as the
    # one for hyperparamter selection

    # TODO: format is a list of dictionary of the type
    # {'metric_name': 'kappa', 'grouping': 'all', 'all': 0.4283837056504599}

    raise NotImplementedError("Not working like expected, scoring values are different ")
    return sum(scores.values())


def build_query(group, unformated_string):
    # many of the columns contain the information 
    # which we extract for formatting part of the instructions
    # given the templates
    # we need a copy here 
    info_source = group.iloc[0].to_dict()
    info_source["criteria"] = "\n".join(f"({i}): {c}" for i, c in 
                                        enumerate(group["grading_criteria"]))
    return format_string(DotMap(info_source), unformated_string)
                  
    
def build_answer(group, unformatted_string):
    info_source = group.iloc[0].to_dict()  # contains string formatting information
    yesno = lambda n: "Yes" if n else "No"
    pairs = group[["grading_criteria", "grading_value"]].to_numpy()
    info_source["answer"] = "\n".join([f"({i}): {yesno(v)}" for i, (c, v) in enumerate(pairs)])
    return format_string(DotMap(info_source), unformatted_string)

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

def match_criteria(beacons, response):
    results = []
    for beacon in beacons:
        # TODO: need to handle the special case where no answer is provided
        # in that case simply return special value which will be mapped
        # to the opposite of what is expected 
        if beacon not in response:
            warn(f"Beacon {beacon} not in response: {response}")
            results.append(None)
            continue

        search_str = response[response.index(beacon) + len(beacon):]
        match_yes = bool(re.match("[^a-zA-Z\d]*Yes", search_str))
        match_no = bool(re.match("[^a-zA-Z\d]*No", search_str))
        if (not match_yes and not match_no) or (match_yes and match_no):
            warn("Model answer does not match expectation")
            results.append(None)
            continue

        results.append(match_yes)

    return results



