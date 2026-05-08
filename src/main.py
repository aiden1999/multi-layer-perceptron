import argparse
import math
import numpy as np

from ingest_data import get_datasets


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
    sum_output = 0
    u_hidden = np.zeros((num_nodes, 1))
    u_output = 0
    f_prime_output = 0
    f_prime_hidden = np.zeros((num_nodes, 1))
    delta_output = 0
    delta_hidden = np.zeros((num_nodes, 1))
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

    while repeat:
        epoch_count += 1
        print(epoch_count)

        # Training
        for k in range(num_rows_training):
            correctOutput = float(training_data[k, num_inputs])

            # Forward Pass
            sum_output = 0
            for i in range(num_nodes):
                sum_hidden[i] = 0
                for j in range(num_inputs):
                    sum_hidden[i] += (
                        float(training_data[k, j]) * weights_input_hidden[i, j]
                    )
                sum_hidden[i] += bias_hidden[i]
                u_hidden[i] = 1 / (1 + np.exp(-sum_hidden[i]))
                sum_output += u_hidden[i] * weights_hidden_output[i]
            sum_output += bias_output
            u_output = 1 / (1 + np.exp(-sum_output))

            # Backward Pass
            f_prime_output = u_output * (1 - u_output)
            delta_output = (correct_output - u_output) * f_prime_output
            for i in range(num_nodes):
                f_prime_hidden[i] = u_hidden[i] * (1 - u_hidden[i])
                delta_hidden[i] = (
                    weights_hidden_output[i] * delta_output * f_prime_hidden[i]
                )

            # Update weights and biases

            # With momentum
            if use_momentum and (epoch_count != 1):
                for i in range(num_nodes):
                    for j in range(num_inputs):
                        weight_diff = (
                            weights_input_hidden[i, j] - weights_input_hidden_old[i, j]
                        )
                        weights_input_hidden_old[i, j] = weights_input_hidden[i, j]
                        weights_input_hidden[i, j] += (
                            step_size * delta_hidden[i] * float(training_data[k, j])
                        ) + (alpha * weight_diff)
                    weight_diff = (
                        weights_hidden_output[i] - weights_hidden_output_old[i]
                    )
                    weights_hidden_output_old[i] = weights_hidden_output[i]
                    weights_hidden_output[i] += +(
                        step_size * delta_output * u_hidden[i]
                    ) + (alpha * weight_diff)
                    bias_diff = bias_hidden[i] - bias_hidden_old[i]
                    bias_hidden_old[i] = bias_hidden[i]
                    bias_hidden[i] += (step_size * delta_hidden[i]) + (
                        alpha * bias_diff
                    )
                bias_diff = bias_output - bias_output_old
                bias_output_old = bias_output
                bias_output += (step_size * delta_output) + (alpha * bias_diff)

            # without momentum
            else:
                for i in range(num_nodes):
                    for j in range(num_inputs):
                        if use_momentum:
                            weights_input_hidden_old[i, j] = weights_input_hidden[i, j]
                        weights_input_hidden[i, j] += (
                            step_size * delta_hidden[i] * float(training_data[k, j])
                        )
                    if use_momentum:
                        weights_hidden_output_old[i] = weights_hidden_output[i]
                        bias_hidden_old[i] = bias_hidden[i]
                    weights_hidden_output[i] += step_size * delta_output * u_hidden[i]
                    bias_hidden[i] += step_size * delta_hidden[i]
                if use_momentum:
                    bias_output_old = bias_output
                bias_output += step_size * delta_output

        # RMSE for training data
        rmse_training_sum = 0
        for k in range(num_rows_training):
            correct_output = float(training_data[k, num_inputs])
            sum_output = 0
            for i in range(num_nodes):
                sum_hidden[i] = 0
                for j in range(num_inputs):
                    sum_hidden[i] += (
                        float(training_data[k, j]) * weights_input_hidden[i, j]
                    )
                sum_hidden[i] += bias_hidden[i]
                u_hidden[i] = 1 / (1 + np.exp(-sum_hidden[i]))
                sum_output += u_hidden[i] * weights_hidden_output[i]
            sum_output += bias_output
            u_output = 1 / (1 + np.exp(-sum_output))
            rmse_training_sum += ((correct_output - u_output) ** 2) / num_rows_training
        rmse_training = np.sqrt(rmse_training_sum)
        rmse_training_file.write(str(float(rmse_training)) + "\n")

        # Passing Validation Data every 5 epochs
        if epoch_count % 5 == 0:
            rmse_validation_sum = 0
            if epoch_count != 5:
                rmse_validation_old = rmse_validation
            for k in range(num_rows_validation):
                correctOutput = float(validation_data[k, num_inputs])
                sumOutput = 0
                for i in range(num_nodes):
                    sum_hidden[i] = 0
                    for j in range(num_inputs):
                        sum_hidden[i] += (
                            float(validation_data[k, j]) * weights_input_hidden[i, j]
                        )
                    sum_hidden[i] += bias_hidden[i]
                    u_hidden[i] = 1 / (1 + np.exp(-sum_hidden[i]))
                    sum_output += u_hidden[i] * weights_hidden_output[i]
                sumOutput += bias_output
                u_output = 1 / (1 + np.exp(-sum_output))
                rmse_validation_sum += (
                    (correct_output - u_output) ** 2
                ) / num_rows_validation
            rmse_validation = np.sqrt(rmse_validation_sum)
            rmse_validation_file.write(str(float(rmse_validation)))
            print("validation " + str(rmse_validation))
            if (rmse_validation > rmse_validation_old) or (epoch_count == 10000):
                repeat = False
        rmse_validation_file.write("\n")

    # Passing Test Data
    for k in range(num_rows_testing):
        correctOutput = float(testing_data[k, num_inputs])
        sum_output = 0
        rmse_testing_sum = 0
        for i in range(num_nodes):
            sum_hidden[i] = 0
            for j in range(num_inputs):
                sum_hidden[i] += float(testing_data[k, j]) * weights_input_hidden[i, j]
            sum_hidden[i] += +bias_hidden[i]
            u_hidden[i] = 1 / (1 + math.exp(-sum_hidden[i]))
            sum_output += u_hidden[i] * weights_hidden_output[i]
        sum_output += bias_output
        u_output = 1 / (1 + np.exp(-sum_output))
        rmse_testing_sum += ((correct_output - u_output) ** 2) / num_rows_testing
        rmse_testing = np.sqrt(rmse_testing_sum)
        rmse_testing_file.write(str(rmse_testing) + "\n")
        output_file.write(str(float(correctOutput)) + " " + str(float(u_output)) + "\n")
        print("test " + str(rmse_testing))

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
