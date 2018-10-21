# # FAPS PLMAgents
# ## FAPS PLM ML-Agent Learning

import logging

import os
import docopt

from FAPSPLMAgents.trainer_controller import TrainerController

if __name__ == '__main__':
    logger = logging.getLogger("FAPSPLMAgents")

    _USAGE = '''
    Usage:
      main [options]
      main --help
    
    Options:
      --keep-checkpoints=<n>     How many model checkpoints to keep [default: 5].
      --lesson=<n>               Start learning from this lesson [default: 0].
      --load                     Whether to load the model or randomly initialize [default: True].
      --run-id=<path>            The sub-directory name for model and summary statistics [default: DQN]. 
      --save-freq=<n>            Frequency at which to save model [default: 10000].
      --seed=<n>                 Random seed used for training [default: -1].
      --train                    Whether to train model, or only run inference [default: False].
      --worker-id=<n>            Number to add to communication port (5005). Used for multi-environment [default: 0].
      --use-gpu                  Make use of GPU.
    '''
    options = None
    try:
        options = docopt.docopt(_USAGE)

    except docopt.DocoptExit as e:
        # The DocoptExit is thrown when the args do not match.
        # We print a message to the user and the usage block.

        print('Invalid Command!')
        print(e)
        exit(1)

    # General parameters
    run_id = options['--run-id']
    seed = int(options['--seed'])
    load_model = options['--load']
    train_model = options['--train']
    save_freq = int(options['--save-freq'])
    keep_checkpoints = int(options['--keep-checkpoints'])
    worker_id = int(options['--worker-id'])
    lesson = int(options['--lesson'])
    use_gpu = int(options['--use-gpu'])

    # log the configuration
    # logger.info(options)

    # Constants
    # Assumption that this yaml is present in same dir as this file
    base_path = os.path.dirname(__file__)
    TRAINER_CONFIG_PATH = os.path.abspath(os.path.join(base_path, "trainer_config.yaml"))

    TrainerController.serve(use_gpu, run_id, save_freq, load_model, train_model, worker_id, keep_checkpoints, lesson,
                            seed, TRAINER_CONFIG_PATH)
    exit(0)
