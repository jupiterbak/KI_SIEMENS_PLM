import logging
import numpy as np
import os
import re
import tensorflow as tf
import keras
import yaml
import keras.backend as B

from concurrent import futures
import time
import grpc

from FAPSPLMAgents.communicatorapi_python import FAPSPLMServives_pb2_grpc as FAPSPLMServivesgrpc

from FAPSPLMAgents.communicatorapi_python import handle_type_proto_pb2 as HandleTypeProto
from FAPSPLMAgents.communicatorapi_python import academy_configuration_proto_pb2 as academy_configuration_proto_pb2
from FAPSPLMAgents.communicatorapi_python import academy_action_proto_pb2 as academy_action_proto_pb2
from FAPSPLMAgents.communicatorapi_python import academy_state_proto_pb2 as academy_state_proto_pb2
from FAPSPLMAgents.communicatorapi_python import action_type_proto_pb2 as action_type_proto_pb2
from FAPSPLMAgents.BrainInfo import BrainInfo as BrainInfo

from FAPSPLMAgents.exception import FAPSPLMEnvironmentException, FAPSPLMActionException

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainerController(FAPSPLMServivesgrpc.FAPSPLMServicesServicer):
    """Provides methods that implement functionality of the FAPSPLMServicesServicer server."""

    def __init__(self, use_gpu, run_id, save_freq, load, train, worker_id, keep_checkpoints, lesson, seed,
                 trainer_config_path):
        """
        :param brain_name: Name of the brain to train
        :param run_id: The sub-directory name for model and summary statistics
        :param save_freq: Frequency at which to save model
        :param load: Whether to load the model or randomly initialize
        :param train: Whether to train model, or only run inference
        :param worker_id: Number to add to communication port (5005). Used for multi-environment
        :param keep_checkpoints: How many model checkpoints to keep
        :param lesson: Start learning from this lesson
        :param seed: Random seed used for training
        :param trainer_config_path: Fully qualified path to location of trainer configuration file
        """
        self.serviser = FAPSPLMServivesgrpc.FAPSPLMServicesServicer()
        self.use_gpu = use_gpu
        self.trainer_config_path = trainer_config_path
        self.logger = logging.getLogger("FAPSPLMAgents")
        self.run_id = run_id
        self.save_freq = save_freq
        self.lesson = lesson
        self.load_model = load
        self.train_model = train
        self.worker_id = worker_id
        self.keep_checkpoints = keep_checkpoints

        # Environment holders
        self.trainer_parameters_dict = {}
        self.trainers = {}
        self.academies = {}
        self.environments_2_trainers_mapping = {}

        # Initialize
        if seed == -1:
            seed = np.random.randint(0, 999999)
        self.seed = seed
        np.random.seed(self.seed)
        if B.backend() == 'tensorflow':
            tf.set_random_seed(self.seed)
        else:
            np.random.seed(seed)
        self._configure()
        self.trainer_config = self._load_config()

    def FAPSAGENT_Initialize(self, request, context):
        academy_request = academy_configuration_proto_pb2.AcademyConfigProto()
        academy_request.CopyFrom(request)

        # Create an environment object with this academy configuration
        academy_name = re.sub('[^0-9a-zA-Z]+', '-', request.AcademyName)
        academy_config = self.academies.get(academy_name)
        if academy_config is None:
            self.academies[academy_name] = academy_request
        else:
            if not academy_config.__eq__(academy_request):
                context.set_code(grpc.StatusCode.ALREADY_EXISTS)
                context.set_details('Academy is already initialized and have a different setting!')
                return HandleTypeProto.HandleTypeProto(handle=-1)
            # indexer = 0
            # while not academy_config.__eq__(academy_request):
            #     new_academy_name = '{}{}'.format(academy_request.AcademyName, indexer)
            #     academy_request.AcademyName = new_academy_name
            #     academy_name = new_academy_name
            #     if self.academies.get([new_academy_name]) is None:
            #         self.academies[new_academy_name] = academy_request
            #     indexer = indexer + 1
            #     academy_config = self.academies.get([academy_name])

        # Get or initialize the trainer mapping
        mapping = self.environments_2_trainers_mapping.get(academy_name)
        if mapping is None:
            # self.environments_2_trainers_mapping[academy_name] = {}
            mapping = {}

        # Initialize the brains and the corresponding trainers
        for i in range(academy_request.brainCount):
            brain_name = re.sub('[^0-9a-zA-Z]+', '-', academy_request.BrainParameter[i].brainName)
            # check if brain already exists
            brain = mapping.get(brain_name)
            if brain is None:
                brain = self.trainers.get(brain_name)
                if brain is None:
                    # Create the model path
                    model_path = 'models/%s' % brain_name
                    self._create_model_path(model_path)
                    # initialize the trainer
                    self._initialize_trainer(academy_request, brain_name, self.trainer_config)
                    brain = self.trainers[brain_name]
                mapping[brain_name] = brain

        # update the trainer mapping
        self.environments_2_trainers_mapping[academy_name] = mapping

        # get the handle Index
        handle = list(self.environments_2_trainers_mapping.keys()).index(academy_name)

        # returm the OK message
        logger.info("\n'{}' started successfully!".format(academy_name))
        context.set_code(grpc.StatusCode.OK)
        return HandleTypeProto.HandleTypeProto(handle=handle)

    def FAPSAGENT_Clear(self, request, context):
        handle = request.handle
        academy_name = list(self.environments_2_trainers_mapping.keys())[handle]
        if academy_name is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Academy with the handle {} was not found!'.format(handle))
            return HandleTypeProto.HandleTypeProto(handle=-1)

        self.environments_2_trainers_mapping[academy_name] = None
        self.academies[academy_name] = None

        context.set_code(grpc.StatusCode.OK)
        return HandleTypeProto.HandleTypeProto(handle=-1)

    def FAPSAGENT_Start(self, request, context):
        handle = request.handle
        academy_name = list(self.environments_2_trainers_mapping.keys())[handle]
        if academy_name is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Academy with the handle {} was not found!'.format(handle))
            return HandleTypeProto.HandleTypeProto(handle=-1)

        academy_config = self.academies[academy_name]
        trainers = self.environments_2_trainers_mapping[academy_name]

        print("\n##################################################################################################")
        print("Academy is starting Training...")
        print("Academy Name: {}".format(academy_config.AcademyName))
        print("Backend : {}".format(B.backend()))
        print("Use cpu: {}".format(self.use_gpu))
        iterator = 0
        for k, t in trainers.items():
            print("Trainer({}): {}".format(iterator, t.__str__()))
            iterator = iterator + 1
        print("##################################################################################################")

        # Initialize the trainer
        for k, t in trainers.items():
            if not t.is_initialized():
                t.initialize()
                # Instantiate model parameters from previously saved models
                if self.load_model:
                    print("\nINFO: Loading models ...")
                    model_path = 'models/%s' % k
                    t.load_model_and_restore(model_path)

        # Write the trainers configurations to Tensorboard
        for brain_name, trainer in trainers.items():
            trainer.write_tensorboard_text('Hyperparameters', trainer.parameters)

        context.set_code(grpc.StatusCode.OK)
        return HandleTypeProto.HandleTypeProto(handle=request.handle)

    def FAPSAGENT_Stop(self, request, context):
        handle = request.handle
        academy_name = list(self.environments_2_trainers_mapping.keys())[handle]
        if academy_name is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Academy with the handle {} was not found!'.format(handle))
            return HandleTypeProto.HandleTypeProto(handle=-1)

        context.set_code(grpc.StatusCode.OK)
        return HandleTypeProto.HandleTypeProto(handle=handle)

    def FAPSAGENT_getAction(self, request, context):
        # Clone the request
        academy_state = academy_state_proto_pb2.AcademyStateProto()
        academy_state.CopyFrom(request)

        # Retrieve the trainers and the configurations from the request handle
        handle = academy_state.handle
        academy_name = list(self.environments_2_trainers_mapping.keys())[handle]
        if academy_name is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Academy with the handle {} was not found!'.format(handle))
            return HandleTypeProto.HandleTypeProto(handle=-1)

        academy_config = self.academies[academy_name]
        trainers = self.environments_2_trainers_mapping[academy_name]

        # Compute the action from the trainers
        academy_actions = academy_action_proto_pb2.AcademyActionProto()
        academy_actions.AcademyName = academy_config.AcademyName
        academy_actions.brainCount = academy_config.brainCount
        academy_actions.handle = handle
        for i in range(academy_state.brainCount):
            last_info = self._get_brain_info(academy_config.BrainParameter[i], academy_state.last_states[i])
            curr_info = self._get_brain_info(academy_config.BrainParameter[i], academy_state.states[i])

            brain_name = re.sub('[^0-9a-zA-Z]+', '-', academy_config.BrainParameter[i].brainName)
            trainer = trainers[brain_name]

            trainer_step = trainer.get_step()
            trainer_max_step = trainer.get_max_steps()

            brain_action = academy_actions.actions.add()
            brain_action.brainName = academy_config.BrainParameter[i].brainName

            # check if trainer is globally done
            globally_done = 0
            if self.train_model and trainer_step >= trainer_max_step:
                globally_done = 1

            # add dummy action if trainer is globally done
            if globally_done == 1:
                if academy_config.BrainParameter[i].actionSpaceType == action_type_proto_pb2.action_continuous:
                    for j in range(academy_config.BrainParameter[i].actionSize):
                        brain_action.actions_continous.append(0.0)
                else:
                    for j in range(academy_config.BrainParameter[i].actionSize):
                        brain_action.actions_discrete.append(0)

                # Reset is needed
                brain_action.reset_needed = 1

                # signal global done
                brain_action.isDone = 1
            else:
                # Add experience
                if academy_config.BrainParameter[i].actionSpaceType == action_type_proto_pb2.action_continuous:
                    trainer.add_experiences(last_info, curr_info.last_actions_continuous, curr_info)
                    trainer.process_experiences(last_info, curr_info.last_actions_continuous, curr_info)
                else:
                    trainer.add_experiences(last_info, curr_info.last_action_discrete, curr_info)
                    trainer.process_experiences(last_info, curr_info.last_action_discrete, curr_info)

                # Process experiences and generate statistics
                if trainer.is_ready_update() and self.train_model and trainer.get_step() <= trainer.get_max_steps():
                    # Perform gradient descent with experience buffer
                    trainer.update_model()
                    # Write training statistics.
                    trainer.write_summary()

                # Save the model by the save frequency
                if self.train_model and trainer_step != 0 and trainer_step % self.save_freq == 0  \
                        and trainer_step <= trainer_max_step:
                    model_path = 'models/%s' % brain_name
                    trainer.save_model(model_path)

                # Compute next action vector
                brain_actions_vect = trainer.take_action(curr_info)

                # construct brain action object
                if academy_config.BrainParameter[i].actionSpaceType == action_type_proto_pb2.action_continuous:
                    for j in range(academy_config.BrainParameter[i].actionSize):
                        brain_action.actions_continous.append(brain_actions_vect[j])
                else:
                    for j in range(academy_config.BrainParameter[i].actionSize):
                        brain_action.actions_discrete.append(brain_actions_vect[j])

                # check if reset is needed
                if curr_info.local_done:
                    brain_action.reset_needed = 1
                else:
                    brain_action.reset_needed = 0

                # check if trainer is globally done
                if self.train_model and trainer_step >= trainer_max_step:
                    brain_action.isDone = 1
                else:
                    brain_action.isDone = 0
        return academy_actions

    def _configure(self):
        # configure tensor flow to use 8 cores
        if self.use_gpu:
            if B.backend() == 'tensorflow':
                config = tf.ConfigProto(device_count={"GPU": 1},
                                        intra_op_parallelism_threads=8,
                                        inter_op_parallelism_threads=8,
                                        allow_soft_placement=True)
                keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
            else:
                raise FAPSPLMEnvironmentException("Other backend environment than Tensorflow are nor supported. ")
        else:
            if B.backend() == 'tensorflow':
                config = tf.ConfigProto(device_count={"CPU": 8},
                                        intra_op_parallelism_threads=8,
                                        inter_op_parallelism_threads=8,
                                        allow_soft_placement=True)
                keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
            else:
                raise FAPSPLMEnvironmentException("Other backend environment than Tensorflow are nor supported. ")

    def _load_config(self):
        try:
            with open(self.trainer_config_path) as data_file:
                trainer_config = yaml.load(data_file)
                return trainer_config
        except IOError:
            raise FAPSPLMEnvironmentException("""Parameter file could not be found here {}.
                                            Will use default Hyper parameters"""
                                              .format(self.trainer_config_path))
        except UnicodeDecodeError:
            raise FAPSPLMEnvironmentException("There was an error decoding Trainer Config from this path : {}"
                                              .format(self.trainer_config_path))

    def _initialize_trainer(self, academy, brain_name, trainer_config):
        trainer_parameters = trainer_config['default'].copy()
        graph_scope = re.sub('[^0-9a-zA-Z]+', '-', brain_name)
        trainer_parameters['graph_scope'] = graph_scope
        trainer_parameters['summary_path'] = '{basedir}/{name}'.format(
            basedir='summaries',
            name=str(self.run_id) + '_' + graph_scope)
        if brain_name in trainer_config:
            _brain_key = brain_name
            while not isinstance(trainer_config[_brain_key], dict):
                _brain_key = trainer_config[_brain_key]
            for k in trainer_config[_brain_key]:
                trainer_parameters[k] = trainer_config[_brain_key][k]
        self.trainer_parameters_dict[brain_name] = trainer_parameters.copy()

        # Instantiate the trainer
        # import the module
        module_spec = self._import_module("FAPSPLMTrainers." + self.trainer_parameters_dict[brain_name]['trainer'],
                                          self.trainer_parameters_dict[brain_name]['trainer'])
        if module_spec is None:
            raise FAPSPLMEnvironmentException("The trainer config contains an unknown trainer type for brain {}"
                                              .format(brain_name))
        else:
            self.trainers[brain_name] = module_spec(academy, brain_name, self.trainer_parameters_dict[brain_name],
                                                    self.train_model, self.seed)

    def _get_brain_info(self, brain_parameter, brain_state):
        return BrainInfo(None, np.asarray(brain_state.states), np.asarray(brain_state.memories),
                         np.asarray(brain_state.rewards), brain_parameter, np.asarray(brain_state.dones),
                         np.asarray(brain_state.last_actions_discrete), np.asarray(brain_state.last_actions_continous))

    def _get_progress(self, brain_name, step_progress, reward_progress):
        """
        Compute and increment the progess of a specified trainer.
        :param brain_name: Name of the brain to train
        :param step_progress: last step
        :param reward_progress: last cummulated reward
        """
        step_progress += self.trainers[brain_name].get_step / self.trainers[brain_name].get_max_steps
        reward_progress += self.trainers[brain_name].get_last_reward
        return step_progress, reward_progress


    @staticmethod
    def _import_module(module_name, class_name):
        """Constructor"""
        module = __import__(module_name)
        my_class = getattr(module, class_name)
        my_class = getattr(my_class, class_name)
        return my_class

    @staticmethod
    def _create_model_path(model_path):
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except Exception:
            raise FAPSPLMEnvironmentException("The folder {} containing the generated model could not be accessed. "
                                              "Please make sure the permissions are set correctly.".format(model_path))

    @staticmethod
    def serve(use_gpu, run_id, save_freq, load, train, worker_id, keep_checkpoints, lesson, seed,
              trainer_config_path):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        FAPSPLMServivesgrpc.add_FAPSPLMServicesServicer_to_server(
            TrainerController(use_gpu, run_id, save_freq, load, train, worker_id,
                              keep_checkpoints, lesson, seed, trainer_config_path)
            , server)
        server.add_insecure_port('[::]:6005')
        server.start()
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)