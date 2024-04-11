import os
from src.data.ICER import ICER
from src.model.Agent import Agent
from src.model.ConversationalAgent import ConversationalAgent
from src.model.OpenAI import ChatGPT
from src.model.llama2 import Llama2
from src.utils.files import create_dir, save_json 

class Experiment():

    def __init__(self, name, config, test_run=False) -> None:
        self.name = name + "_test_run" if test_run else name
        self.config = config 
        self.test_run = test_run
        
        self._init_directories()
        self.dataset_handler = self._load_dataset_handler()
        self.agent = self._load_agent()

    def _init_directories(self):
        if self.test_run: self.config.name = self.config.name + "_test_run"
        self.save_dir = os.path.join(self.config.save_dir, self.config.name)
        create_dir(self.save_dir)
        self.results_save_dir = os.path.join(self.save_dir, "results")
        create_dir(self.results_save_dir)
        # let's save the configuration used 
        save_json(self.config, os.path.join(self.save_dir, 
                                            "configuration.json"))

    def _load_dataset_handler(self):
        """ 
        Instantiate the object responsible for processing the
        dataset and handling evaluation functionalities. 
        """

        ds_name = self.config.dataset.name
        if ds_name.startswith("icer"):
            return ICER(self.config.dataset)
        else:
            raise ValueError(f"Unknown dataset {ds_name}")
        
    def _load_agent(self):
        """ 
        Instantiate the object responsible for handling
        the LLM loading and querying functionalities.
        """

        conf = self.config.agent
        
        if conf.name.startswith("gpt") and conf.source != "huggingface":
            return ChatGPT(self.config.agent, seed=self.config.seed)
        elif conf.source == "huggingface" and conf.is_chat:
            return ConversationalAgent(self.config.agent)
        elif conf.source == "huggingface" and not conf.is_chat:
            if "llama" in self.config.name:
                return Llama2(self.config.agent)
            return Agent(self.config.agent)
        else:
            raise ValueError(f"Unkwon agent type {conf}")
        
    
    def _save_results(self, results, gen_param, save_dir, suffix=""):
        name = "top_p_{top_p}_temp_{temp}_n_{n}_{suffix}.json".format(
            top_p=gen_param["top_p"], temp=gen_param["temperature"], 
            n=gen_param["n"], suffix=suffix)
        results_save_path = os.path.join(save_dir, name)
        save_json(results, results_save_path)  
