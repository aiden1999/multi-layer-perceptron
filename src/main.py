import argparse
import numpy as np

from ingest_data import get_datasets
from testing import test
from training import train


def main(*, num_nodes=6, step_size=0.1, use_momentum=True):
    training_data, validation_data, testing_data = get_datasets()
    num_rows_training, num_cols = np.shape(training_data)
    num_rows_validation = np.shape(validation_data)[0]
    num_rows_testing = np.shape(testing_data)[0]

    # Create files for RMSE
    rmse_training_file = open("trainingRMSE.txt", "w")
    rmse_validation_file = open("validationRMSE.txt", "w")
    rmse_testing_file = open("testRMSE.txt", "w")
    output_file = open("output.txt", "w")

    num_inputs = num_cols - 1

    correct_output = 0

    weights_input_hidden = np.zeros((num_nodes, num_inputs))
    bias_hidden = np.zeros((num_nodes, 1))
    weights_hidden_output = np.zeros((num_nodes, 1))

    lower_inputs = -2 / num_inputs
    upper_inputs = 2 / num_inputs
    lower_nodes = -2 / num_nodes
    upper_nodes = 2 / num_nodes
    for i in range(num_nodes):
        for j in range(num_inputs):
            weights_input_hidden[i, j] = np.random.uniform(lower_inputs, upper_inputs)
        bias_hidden[i] = np.random.uniform(lower_inputs, upper_inputs)
        weights_hidden_output[i] = np.random.uniform(lower_nodes, upper_nodes)
    bias_output = np.random.uniform(lower_nodes, upper_nodes)
    sum_hidden = np.zeros((num_nodes, 1))
    u_hidden = np.zeros((num_nodes, 1))
    u_output = 0
    f_prime_output = 0
    f_prime_hidden = np.zeros((num_nodes, 1))
    delta_output = 0
    rmse_validation_sum = 0
    rmse_testing_sum = 0
    epoch_count = 0
    repeat = True
    rmse_validation_old = 100
    alpha = 0.9
    weights_input_hidden_old = np.zeros((num_nodes, num_inputs))
    weights_hidden_output_old = np.zeros((num_nodes, 1))
    bias_hidden_old = np.zeros((num_nodes, 1))
    bias_output_old = 0

    train(training_data, num_nodes, num_inputs)
    test()

    rmse_training_file.close()
    rmse_validation_file.close()
    rmse_testing_file.close()
    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--momentum", type=bool, default=True)

    args = parser.parse_args()
    main(num_nodes=args.nodes, step_size=args.step_size, use_momentum=args.momentum)
