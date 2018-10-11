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

from FAPSPLMAgentRPC.communicatorapi_python import FAPSPLMServives_pb2_grpc as FAPSPLMServivesgrpc
from FAPSPLMAgentRPC.communicatorapi_python import FAPSPLMServives_pb2 as FAPSPLMServives

from .brain import BrainInfo, BrainParameters
from .exception import FAPSPLMEnvironmentException, FAPSPLMActionException

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAPSPLMEnvironmentRPC(FAPSPLMServivesgrpc.FAPSPLMServicesServicer):
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
        self.trainer_parameters_dict = {}
        self.trainers = {}
        if seed == -1:
            seed = np.random.randint(0, 999999)
        self.seed = seed
        np.random.seed(self.seed)
        if B.backend() == 'tensorflow':
            tf.set_random_seed(self.seed)
        else:
            np.random.seed(seed)
        self._configure()

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
        self.trainer_config = self._load_config()

    def _initialize_trainer(self, brain_name, trainer_config):
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
            self.trainers[brain_name] = module_spec(self.env, brain_name, self.trainer_parameters_dict[brain_name],
                                                    self.train_model, self.seed)

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

    def _save_model(self):
        """
        Saves current model to checkpoint folder.
        """
        for k, t in self.trainers.items():
            t.save_model(self.model_path)
        print("\nINFO: Model saved.")

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

    def FAPSAGENT_Initialize(self, request, context):
        # missing associated documentation comment in .proto file
        self.brain_name = re.sub('[^0-9a-zA-Z]+', '-', request.AcademyName)
        self.model_path = 'models/%s' % self.brain_name
        self._create_model_path(self.model_path)

        # Choose and instantiate a trainer
        # TODO: Instantiate the trainer depending on the configurations sent by the
        # environment. Current implementation considers only the brain started on the server side - Jupiter
        self._initialize_trainer(self.brain_name, self.trainer_config)

        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def FAPSAGENT_Clear(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def FAPSAGENT_Start(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def FAPSAGENT_Stop(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def FAPSAGENT_getAction(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def serve(self, use_gpu, run_id, save_freq, load, train, worker_id, keep_checkpoints, lesson, seed, trainer_config_path):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        FAPSPLMServivesgrpc.add_FAPSPLMServicesServicer_to_server(
            FAPSPLMEnvironmentRPC( use_gpu, run_id, save_freq, load, train, worker_id,
                                   keep_checkpoints, lesson, seed, trainer_config_path)
            , server)
        server.add_insecure_port('[::]:6005')
        server.start()
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)
