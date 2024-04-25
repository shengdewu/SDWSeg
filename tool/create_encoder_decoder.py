import torch
from importlib import import_module
import shutil
import sys
import types
import tempfile
import os

from engine.model.build import build_network

from sdwseg.model import EncoderDecoder


def load_config(file_name):
    cfg_dict = dict()
    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(
            dir=temp_config_dir, suffix='.py')
        temp_config_name = os.path.basename(temp_config_file.name)
        shutil.copyfile(file_name, temp_config_file.name)
        temp_module_name = os.path.splitext(temp_config_name)[0]
        sys.path.insert(0, temp_config_dir)
        mod = import_module(temp_module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
               and not isinstance(value, types.ModuleType)
               and not isinstance(value, types.FunctionType)
        }
        del sys.modules[temp_module_name]
        temp_config_file.close()
    return cfg_dict


def create_encoder_decoder(cfg_path: str, checkpoint: str = '') -> EncoderDecoder:
    config = load_config(cfg_path)
    model = build_network(config['model'])
    if checkpoint != '':
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict['model']['g_model'])
    model.preparate_deploy()
    model.eval()
    return model
