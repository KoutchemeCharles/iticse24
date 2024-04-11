from src.Experiment import Experiment

class Grading(Experiment):

    def __init__(self, config, test_run=False) -> None:
        super().__init__("grading", config, test_run)

    @classmethod
    def only_for_grading(cls, config):
        cls._init_directories = lambda x: None
        cls._load_dataset_handler = lambda x: None 
        cls._generate_and_evaluate = None 
        cls._evaluate = None
        
        return cls(config)

    from .grade import (
        run, 
        _zero_shot_prompt, 
        _few_shot_prompt, 
        create_zero_shot_messages,
        create_few_shot_messages,
        query_and_broadcast,
        _generate, _evaluate,
        _generate_and_evaluate,
        search_best_gen_params,
        compute_objective
    )

    from .score import compute_scores

