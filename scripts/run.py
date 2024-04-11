from argparse import ArgumentParser
from src.grading.Grading import Grading
from src.feedback.Feedback import Feedback
from src.utils.core import set_seed
from src.utils.files import read_config


def parse_args():
    parser = ArgumentParser(description="Running experiments")
    parser.add_argument("--config", required=True,
                        help="Path towards the configuration file")
    parser.add_argument('--test_run',
                        help="Whether to do a test run to ensure the pipeline works without issues",
                        action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    config = read_config(args.config)
    set_seed(config.seed)

    if config.task.task == "grading":
        experiment = Grading(config, test_run=args.test_run) # TODO: rename into Judge
    elif config.task.task == "feedback":
        experiment = Feedback(config, test_run=args.test_run)
    else:
        raise ValueError(f"Experiment {args.experiment} not implemented")
    
    experiment.run()


if __name__ == "__main__":
    main()