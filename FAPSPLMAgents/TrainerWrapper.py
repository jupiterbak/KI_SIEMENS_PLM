import logging
import queue
import threading
import time

from FAPSPLMAgents.communicatorapi_python import brain_action_proto_pb2 as brain_action_proto_pb2
from FAPSPLMAgents.communicatorapi_python import action_type_proto_pb2 as action_type_proto_pb2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainerWrapper:
    def __init__(self, trainer_name, trainer_controller, trainer):
        self.actionInputQueue = queue.Queue()
        self.actionOutputQueue = queue.Queue()
        self.thread = TrainerThread(trainer_name, trainer_controller, trainer, self.actionInputQueue,
                                    self.actionOutputQueue)

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()
            self.thread.get_event_ready().wait()

    def stop(self):
        if self.thread.is_alive():
            self.thread.stop()
            self.thread.join()
            self.thread = None

    def get_action(self, brain_action, brain_parameter, last_info, curr_info):
        self.actionInputQueue.put([brain_action, brain_parameter, last_info, curr_info])
        data = self.actionOutputQueue.get()
        self.actionOutputQueue.task_done()
        return data

    def __str__(self):
        return self.thread.trainer.__str__()


class TrainerThread(threading.Thread):
    def __init__(self, trainer_name, trainer_controller, trainer, action_input_queue, action_output_queue):
        threading.Thread.__init__(self)
        self.trainer_name = trainer_name
        self.trainerController = trainer_controller
        self.trainer = trainer
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self.actionInputQueue = action_input_queue
        self.actionOutputQueue = action_output_queue

    def get_event_ready(self):
        return self._ready_event

    def set_ready(self):
        self._ready_event.set()

    def is_ready(self):
        return self._ready_event.is_set()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        # Initialize the trainer
        if not self.trainer.is_initialized():
            self.trainer.initialize()
            # Instantiate model parameters from previously saved models
            if self.trainerController.load_model:
                    print("\nINFO: Loading models ...")
                    model_path = 'models/%s' % self.trainer_name
                    self.trainer.load_model_and_restore(model_path)

        # Write the trainers configurations to Tensorboard
            self.trainer.write_tensorboard_text('Hyperparameters', self.trainer.parameters)

        self.set_ready()

        # Main Trainer Loop
        while not self.stopped():
            [brain_action, brain_parameter, last_info, curr_info] = self.actionInputQueue.get()

            if (brain_parameter is None) or (last_info is None) or (curr_info is None):
                time.sleep(1)
            else:

                brain_action.brainName = brain_parameter.brainName

                trainer_step = self.trainer.get_step
                trainer_max_step = self.trainer.get_max_steps

                # check if trainer is globally done
                globally_done = 0
                if self.trainerController.train_model and trainer_step >= trainer_max_step:
                    globally_done = 1

                self.trainer.increment_step()
                # add dummy action if trainer is globally done
                if globally_done == 1:
                    if brain_parameter.actionSpaceType == action_type_proto_pb2.action_continuous:
                        for j in range(brain_parameter.actionSize):
                            brain_action.actions_continous.append(0.0)
                    else:
                        for j in range(brain_parameter.actionSize):
                            brain_action.actions_discrete.append(0)
                    self.trainer.end_episode()

                    # Reset is needed
                    brain_action.reset_needed = 1

                    # signal global done
                    brain_action.isDone = 1
                else:
                    # Add experience
                    if brain_parameter.actionSpaceType == action_type_proto_pb2.action_continuous:
                        self.trainer.add_experiences(last_info, curr_info.last_actions_continuous, curr_info)
                        self.trainer.process_experiences(last_info, curr_info.last_actions_continuous, curr_info)
                    else:
                        self.trainer.add_experiences(last_info, curr_info.last_action_discrete, curr_info)
                        self.trainer.process_experiences(last_info, curr_info.last_action_discrete, curr_info)

                    # Process experiences and generate statistics
                    if self.trainer.is_ready_update() and self.trainerController.train_model \
                            and self.trainer.get_step <= self.trainer.get_max_steps:
                        # Perform gradient descent with experience buffer
                        self.trainer.update_model()
                        # Write training statistics.
                        self.trainer.write_summary()

                    # Save the model by the save frequency
                    if self.trainerController.train_model and trainer_step != 0 \
                            and trainer_step % self.trainerController.save_freq == 0 \
                            and trainer_step <= trainer_max_step:
                        model_path = 'models/%s' % self.trainer_name
                        self.trainer.save_model(model_path)

                    # Compute next action vector
                    brain_actions_vect = self.trainer.take_action(curr_info)

                    # construct brain action object
                    if brain_parameter.actionSpaceType == action_type_proto_pb2.action_continuous:
                        for j in range(brain_parameter.actionSize):
                            brain_action.actions_continous.append(brain_actions_vect[j])
                    else:
                        for j in range(brain_parameter.actionSize):
                            brain_action.actions_discrete.append(brain_actions_vect[j])

                    # check if reset is needed
                    if curr_info.local_done:
                        brain_action.reset_needed = 1
                    else:
                        brain_action.reset_needed = 0

                    # check if trainer is globally done
                    if self.trainerController.train_model and trainer_step >= trainer_max_step:
                        brain_action.isDone = 1
                    else:
                        brain_action.isDone = 0

                self.actionOutputQueue.put(brain_action)
            time.sleep(0.5)
            self.actionInputQueue.task_done()
