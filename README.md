# Open Source Language Models Can Provide Feedback

Repository for the paper "Open Source Language Models Can Provide Feedback: Evaluating LLMs’ Ability to Help Students Using GPT-4-As-A-Judge" published at ITICSE24.

This repository provides the code used to produce the feedback and evaluating the grading. 
Because the data used in the study is private, this code is not *reproducable*.
It only serves as an illustration of the pipeline applied in the paper. 

If you have inquiries about technical details to run similar experiments, 
please contact the authors at **charles.koutcheme@aalto.fi**.


### Example running

Move to the directory containing the code and run the following code:

module load miniconda
source activate feedeval;

```python
python scripts/run.py --config ./configs/experiments/feedback_iticse/example.json
```

