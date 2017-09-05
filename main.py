"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""

import json
import logging
import os
import numpy as np
import tensorflow as tf
from config import get_config
from equation import get_equation
from solver import FeedForwardModel

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('problem_name', 'HJB',
                           """The name of partial differential equation.""")
tf.app.flags.DEFINE_integer('num_run', 1,
                            """The number of experiments to repeatedly run for the same problem.""")
tf.app.flags.DEFINE_string('log_dir', './logs',
                           """Directory where to write event logs and output array.""")


def main():
    problem_name = FLAGS.problem_name
    config = get_config(problem_name)
    bsde = get_equation(problem_name, config.dim, config.total_time, config.num_time_interval)

    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    path_prefix = os.path.join(FLAGS.log_dir, problem_name)
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(dict((name, getattr(config, name))
                       for name in dir(config) if not name.startswith('__')),
                  outfile, indent=2)
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)-6s %(message)s')

    for idx_run in range(1, FLAGS.num_run+1):
        tf.reset_default_graph()
        with tf.Session() as sess:
            logging.info('Begin to solve %s with run %d' % (problem_name, idx_run))
            model = FeedForwardModel(config, bsde, sess)
            if bsde.y_init:
                logging.info('Y0_true: %.4e' % bsde.y_init)
            model.build()
            training_history = model.train()
            if bsde.y_init:
                logging.info('relative error of Y0: %s',
                             '{:.2%}'.format(
                                 abs(bsde.y_init - training_history[-1, 2])/bsde.y_init))
            # save training history
            np.savetxt('{}_training_history_{}.csv'.format(path_prefix, idx_run),
                       training_history,
                       fmt=['%d', '%.5e', '%.5e', '%d'],
                       delimiter=",",
                       header="step,loss_function,target_value,elapsed_time",
                       comments='')

if __name__ == '__main__':
    main()
