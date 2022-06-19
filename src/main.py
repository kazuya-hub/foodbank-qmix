import numpy as np
import os
# import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run

# set to "no" if you want to see stdout/stderr in console
SETTINGS['CAPTURE_MODE'] = "fd"
logger = get_logger()

# Sacredで実験を定義
ex = Experiment("foodbank_qmix")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


# Sacredで実験を実行すると最初に呼び出される
@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    # config = config_copy(_config)
    np.random.seed(_config["seed"])
    th.manual_seed(_config["seed"])
    _config['env_args']['seed'] = _config["seed"]

    # run the framework
    run(_run, _config, _log)


if __name__ == '__main__':
    # 実行時の引数
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    # default.yamlからデフォルトのパラメータを読み込む
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
            print("Loaded config yaml")
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # now add all the config to sacred
    # Scaredにパラメータを設定
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    # 実験を実行
    ex.run_commandline(params)
