""" 
Base class for interacting with various Language Models
through external APIs. 
"""

class RemoteAgent():
    """
    a fast agent can leverage easyllm to be queried instead of relying on 
    custom gpus.
    """

    def __init__(self, config) -> None:
        self.config = config 

    def query(self, formated_instructions, **gen_kwargs):
        # Get the model generations 

        if self.config.is_chat:
            # TODO: should be single
            responses = self.query_with_message(formated_instructions, 
                                                 **gen_kwargs)
        else:
            prompt = self.build_prompt(formated_instructions)
            responses = self.query_with_prompt(prompt, **gen_kwargs)

        return responses