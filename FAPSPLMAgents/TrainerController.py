import logging
import os
import re
import time
from concurrent import futures

import grpc
import keras
import keras.backend as backend
import numpy as np
import tensorflow as tf
import yaml

from FAPSPLMAgents.BrainInfo import BrainInfo as BrainInfo
from FAPSPLMAgents.TrainerWrapper import TrainerWrapper
from FAPSPLMAgents.communicatorapi_python import FAPSPLMServives_pb2_grpc as FAPSPLMServivesgrpc
from FAPSPLMAgents.communicatorapi_python import academy_action_proto_pb2 as academy_action_proto_pb2
from FAPSPLMAgents.communicatorapi_python import academy_configuration_proto_pb2 as academy_configuration_proto_pb2
from FAPSPLMAgents.communicatorapi_python import academy_state_proto_pb2 as academy_state_proto_pb2
from FAPSPLMAgents.communicatorapi_python import handle_type_proto_pb2 as handle_type_proto_pb2
from FAPSPLMAgents.exception import FAPSPLMEnvironmentException

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainerController(FAPSPLMServivesgrpc.FAPSPLMServicesServicer):
    """Provides methods that implement functionality of the FAPSPLMServicesServicer server."""

    def __init__(self, use_gpu, run_id, save_freq, load, train, worker_id, keep_checkpoints, lesson, seed,
                 trainer_config_path):
        """
        :param use_gpu: Specify to use the GPU
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
        self.servicer = FAPSPLMServivesgrpc.FAPSPLMServicesServicer()
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
        if backend.backend() == 'tensorflow':
            tf.set_random_seed(self.seed)
        else:
            np.random.seed(seed)
        # self._configure()
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
                return handle_type_proto_pb2.HandleTypeProto(handle=-1)
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
            brain_params = academy_request.BrainParameter[i]
            # check if brain already exists
            brain = mapping.get(brain_name)
            if brain is None:
                brain = self.trainers.get(brain_name)
                if brain is None:
                    # Create the model path
                    model_path = 'models/%s' % brain_name
                    self._create_model_path(model_path)
                    # initialize the trainer
                    self._initialize_trainer(brain_params, brain_name, self.trainer_config)
                    brain = self.trainers[brain_name]
                mapping[brain_name] = brain

        # update the trainer mapping
        self.environments_2_trainers_mapping[academy_name] = mapping

        # get the handle Index
        handle = list(self.environments_2_trainers_mapping.keys()).index(academy_name)

        # return the OK message
        logger.info("\n'{}' started successfully!".format(academy_name))
        context.set_code(grpc.StatusCode.OK)
        return handle_type_proto_pb2.HandleTypeProto(handle=handle)

    def FAPSAGENT_Clear(self, request, context):
        # Get the trainer wrappers
        handle = request.handle
        academy_name = list(self.environments_2_trainers_mapping.keys())[handle]
        if academy_name is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Academy with the handle {} was not found!'.format(handle))
            return handle_type_proto_pb2.HandleTypeProto(handle=-1)

        # clear the map entry
        self.environments_2_trainers_mapping[academy_name] = None
        self.academies[academy_name] = None

        # return the OK message
        context.set_code(grpc.StatusCode.OK)
        return handle_type_proto_pb2.HandleTypeProto(handle=-1)

    def FAPSAGENT_Start(self, request, context):
        # get the trainer wrappers
        handle = request.handle
        academy_name = list(self.environments_2_trainers_mapping.keys())[handle]
        if academy_name is None:
            # return not found response
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Academy with the handle {} was not found!'.format(handle))
            return handle_type_proto_pb2.HandleTypeProto(handle=-1)

        academy_config = self.academies[academy_name]
        trainers = self.environments_2_trainers_mapping[academy_name]

        if trainers is None:
            # return not found response
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Academy with the handle {} was not found!'.format(handle))
            return handle_type_proto_pb2.HandleTypeProto(handle=-1)

        # log the academy starting process
        print("\n##################################################################################################")
        print("Academy is starting Training...")
        print("Client: {}".format(context._rpc_event.call_details.host))
        print("Academy Name: {}".format(academy_config.AcademyName))
        print("Backend : {}".format(backend.backend()))
        print("Use cpu: {}".format(self.use_gpu))
        iterator = 0
        for k, t in trainers.items():
            print("Trainer({}): {}".format(iterator, t.__str__()))
            iterator = iterator + 1
        print("##################################################################################################")

        # Start the trainer  wrapper
        for k, t in trainers.items():
            t.start()

        # Return the OK message
        context.set_code(grpc.StatusCode.OK)
        return handle_type_proto_pb2.HandleTypeProto(handle=request.handle)

    def FAPSAGENT_Stop(self, request, context):
        # Get the trainer wrappers
        handle = request.handle
        academy_name = list(self.environments_2_trainers_mapping.keys())[handle]
        if academy_name is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Academy with the handle {} was not found!'.format(handle))
            return handle_type_proto_pb2.HandleTypeProto(handle=-1)

        trainers = self.environments_2_trainers_mapping[academy_name]
        if trainers is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Academy with the handle {} was not found!'.format(handle))
            return handle_type_proto_pb2.HandleTypeProto(handle=-1)

        # Stop the trainer wrappers
        for k, t in trainers.items():
            t.stop()

        context.set_code(grpc.StatusCode.OK)
        return handle_type_proto_pb2.HandleTypeProto(handle=handle)

    def FAPSAGENT_getAction(self, request, context):
        # Clone the request
        academy_state = academy_state_proto_pb2.AcademyStateProto()
        academy_state.CopyFrom(request)

        # Retrieve the trainers and the configurations from the request handle
        handle = academy_state.handle.handle
        academy_name = list(self.environments_2_trainers_mapping.keys())[handle]
        if academy_name is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Academy with the handle {} was not found!'.format(handle))
            return handle_type_proto_pb2.HandleTypeProto(handle=-1)

        academy_config = self.academies[academy_name]
        trainers = self.environments_2_trainers_mapping[academy_name]
        if trainers is None or academy_config is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Academy with the handle {} was not found!'.format(handle))
            return handle_type_proto_pb2.HandleTypeProto(handle=-1)

        # Compute the action from the trainers
        academy_actions = academy_action_proto_pb2.AcademyActionProto()
        academy_actions.AcademyName = academy_config.AcademyName
        academy_actions.brainCount = academy_config.brainCount
        academy_actions.handle.handle = handle
        for i in range(academy_state.brainCount):
            last_info = self._get_brain_last_info(academy_config.BrainParameter[i], academy_state.states[i])
            curr_info = self._get_brain_info(academy_config.BrainParameter[i], academy_state.states[i])
            brain_parameter = academy_config.BrainParameter[i]

            brain_name = re.sub('[^0-9a-zA-Z]+', '-', academy_config.BrainParameter[i].brainName)
            trainer = trainers[brain_name]

            brain_action = trainer.get_action(brain_parameter, last_info, curr_info)
            academy_actions.actions.add(brain_action)

        return academy_actions

    def _configure(self):
        # configure tensor flow to use 8 cores
        if self.use_gpu:
            if backend.backend() == 'tensorflow':
                config = tf.ConfigProto(device_count={"GPU": 1},
                                        intra_op_parallelism_threads=8,
                                        inter_op_parallelism_threads=8,
                                        allow_soft_placement=True)
                keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
            else:
                raise FAPSPLMEnvironmentException("Other backend environment than Tensorflow are not supported. ")
        else:
            if backend.backend() == 'tensorflow':
                config = tf.ConfigProto(device_count={"CPU": 8},
                                        intra_op_parallelism_threads=8,
                                        inter_op_parallelism_threads=8,
                                        allow_soft_placement=True)
                keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
            else:
                raise FAPSPLMEnvironmentException("Other backend environment than Tensorflow are not supported. ")

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
            self.trainers[brain_name] = TrainerWrapper(brain_name, self, module_spec(academy, brain_name,
                                                                                     self.trainer_parameters_dict[
                                                                                         brain_name], self.train_model,
                                                                                     self.seed))

    @staticmethod
    def _get_brain_info(brain_parameter, brain_state):
        return BrainInfo(None, np.asarray(brain_state.states).reshape((1, brain_parameter.stateSize)),
                         np.asarray(brain_state.memories),
                         np.asarray(brain_state.reward), brain_parameter, np.asarray(brain_state.done),
                         np.asarray(brain_state.last_actions_discrete), np.asarray(brain_state.last_actions_continous))

    @staticmethod
    def _get_brain_last_info(brain_parameter, brain_state):
        return BrainInfo(None, np.asarray(brain_state.last_states).reshape((1, brain_parameter.stateSize)),
                         np.asarray(brain_state.memories),
                         np.asarray(brain_state.reward), brain_parameter, np.asarray(brain_state.done),
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
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        FAPSPLMServivesgrpc.add_FAPSPLMServicesServicer_to_server(
            TrainerController(use_gpu, run_id, save_freq, load, train, worker_id,
                              keep_checkpoints, lesson, seed, trainer_config_path), server)
        server.add_insecure_port('[::]:6005')
        server.start()
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)
