# models/SepReformer_Base_WSJ0/main.py
import os
import torch
from loguru import logger
from .dataset import get_dataloaders
from .model import Model
from .engine import Engine
from utils import util_system, util_implement
from utils.decorators import *

# Setup logger
log_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "log/system_log.log"
)
logger.add(log_file_path, level="DEBUG", mode="w")

@logger_wraps()
def main(args):

    ''' Build Setting '''
    # Call configuration file (configs.yaml)
    yaml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs.yaml"
    )
    yaml_dict = util_system.parse_yaml(yaml_path)
    
    # Get full config dict
    config = yaml_dict["config"]
    
    # Call DataLoader [train / valid / test / etc...]
    dataloaders = get_dataloaders(
        args, 
        config["dataset"], 
        config["dataloader"]
    )
    
    ''' Build Model '''
    model = Model(**config["model"])

    ''' Build Engine '''
    # GPU setup
    gpuid = tuple(map(int, config["engine"]["gpuid"].split(',')))
    device = torch.device(f'cuda:{gpuid[0]}')
    
    # Criterion / Optimizer / Scheduler
    criterions = util_implement.CriterionFactory(
        config["criterion"], device
    ).get_criterions()
    optimizers = util_implement.OptimizerFactory(
        config["optimizer"], model.parameters()
    ).get_optimizers()
    schedulers = util_implement.SchedulerFactory(
        config["scheduler"], optimizers
    ).get_schedulers()
    
    # Initialize & run
    engine = Engine(
        args, config, model,
        dataloaders, criterions,
        optimizers, schedulers,
        gpuid, device
    )
    if args.engine_mode == 'infer_sample':
        engine._inference_sample(args.sample_file)
    else:
        engine.run()
