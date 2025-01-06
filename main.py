"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""

import json
import os
import logging

from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np
import tensorflow as tf

import equation as eqn
from solver import BSDESolver


flags.DEFINE_string('config_path', 'configs/hjb_lq_d100.json',
                    """The path to load json file.""")
flags.DEFINE_string('exp_name', 'test',
                    """The name of numerical experiments, prefix for logging""")
FLAGS = flags.FLAGS
FLAGS.log_dir = './logs'  # directory where to write event logs and output array


def main(argv):
    del argv
    tf.keras.backend.clear_session()
    with open(FLAGS.config_path) as json_data_file:
        config_dict = json.load(json_data_file)

    class DictToObject:
        def __init__(self, dictionary):
            self._dict = dictionary
            for key, value in dictionary.items():
                setattr(self, key, value)

        def to_dict(self):
            return self._dict

    class Config:
        def __init__(self, config_dict):
            self.eqn_config = DictToObject(config_dict['eqn_config'])
            self.net_config = DictToObject(config_dict['net_config'])
            self._original_dict = config_dict

        def to_dict(self):
            return self._original_dict

    config = Config(config_dict)
    tf.keras.backend.set_floatx(config.net_config.dtype)

    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    bsde_solver = BSDESolver(config, bsde)

    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    path_prefix = os.path.join(FLAGS.log_dir, FLAGS.exp_name)
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(config.to_dict(), outfile, indent=2)

    absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
    absl_logging.set_verbosity('info')

    logging.info('Begin to solve %s ' % config.eqn_config.eqn_name)
    training_history = bsde_solver.train()
    if bsde.y_init:
        logging.info('Y0_true: %.4e' % bsde.y_init)
        logging.info('relative error of Y0: %s',
                     '{:.2%}'.format(abs(bsde.y_init - training_history[-1, 2])/bsde.y_init))
    np.savetxt('{}_training_history.csv'.format(path_prefix),
               training_history,
               fmt=['%d', '%.5e', '%.5e', '%d'],
               delimiter=",",
               header='step,loss_function,target_value,elapsed_time',
               comments='')

if __name__ == '__main__':

    app.run(main)
