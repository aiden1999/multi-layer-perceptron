import numpy as np


def train(training_data, num_nodes, num_inputs):
    # forward pass
    # backward pass
    # update weights and biases
    # weights: Widrow-Hoff rule and ADALINE
    # for a node with n inputs: [-2/n, 2,n]
    # compute RMSE
    # validate data every n epochs

    weights_input_hidden, weights_hidden_output = calculate_starting_weights(
        num_nodes, num_inputs
    )  # TODO: rename weights variables
    bias_hidden, bias_output = calculate_starting_biases(num_nodes)
    u_hidden = np.zeros((num_nodes, 1))
    delta_hidden = np.zeros((num_nodes, 1))
    repeat = True
    epoch_count = 0
    while repeat:  # FIX: this is nasty
        epoch_count += 1
        print(epoch_count)  # TODO: replace with logging
        for row in training_data:
            correct_output = row[-1]
            u_output = forward_pass(
                row,
                num_nodes,
                num_inputs,
                weights_input_hidden,
                bias_hidden,
                weights_hidden_output,
                u_hidden,
                bias_output,
            )
            backward_pass(
                u_output,
                correct_output,
                num_nodes,
                u_hidden,
                delta_hidden,
                weights_hidden_output,
            )
            update_weights_and_biases()
        calculate_rmse()
        validate()


def calculate_starting_weights(num_nodes, num_inputs):
    weights_input_hidden = np.zeros((num_nodes, num_inputs))
    weights_hidden_output = np.zeros((num_nodes, 1))
    lower_inputs = -2 / num_inputs
    upper_inputs = 2 / num_inputs
    lower_nodes = -2 / num_nodes
    upper_nodes = 2 / num_nodes

    for node in range(num_nodes):
        for input in range(num_inputs):
            weights_input_hidden[node, input] = np.random.uniform(
                lower_inputs, upper_inputs
            )
        weights_hidden_output[node] = np.random.uniform(lower_nodes, upper_nodes)
    return weights_input_hidden, weights_hidden_output


def calculate_starting_biases(num_nodes):
    bias_hidden = np.zeros((num_nodes, 1))
    lower_nodes = -2 / num_nodes
    upper_nodes = 2 / num_nodes

    for node in range(num_nodes):
        bias_hidden[node] = np.random.uniform(lower_nodes, upper_nodes)
    bias_output = np.random.uniform(lower_nodes, upper_nodes)
    return bias_hidden, bias_output


def forward_pass(
    row,
    num_nodes,
    num_inputs,
    weights_input_hidden,
    bias_hidden,
    weights_hidden_output,
    u_hidden,
    bias_output,
):
    sum_output = 0
    sum_hidden = np.zeros((num_nodes, 1))
    for node in range(num_nodes):
        for input in range(num_inputs):
            sum_hidden[node] += row[input] * weights_input_hidden[node, input]
        sum_hidden[node] += bias_hidden[node]
        u_hidden[node] = 1 / (1 + np.exp(-sum_hidden[node]))
        sum_output += u_hidden[node] * weights_hidden_output[node]
    sum_output += bias_output
    u_output = 1 / (1 + np.exp(-sum_output))
    return u_output


def backward_pass(
    u_output, correct_output, num_nodes, u_hidden, delta_hidden, weights_hidden_output
):
    # Backward Pass
    f_prime_output = u_output * (1 - u_output)
    delta_output = (correct_output - u_output) * f_prime_output
    for node in range(num_nodes):
        f_prime_hidden = u_hidden[node] * (1 - u_hidden[node])
        delta_hidden[node] = weights_hidden_output[node] * delta_output * f_prime_hidden


def update_weights_and_biases():

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
            weight_diff = weights_hidden_output[i] - weights_hidden_output_old[i]
            weights_hidden_output_old[i] = weights_hidden_output[i]
            weights_hidden_output[i] += +(step_size * delta_output * u_hidden[i]) + (
                alpha * weight_diff
            )
            bias_diff = bias_hidden[i] - bias_hidden_old[i]
            bias_hidden_old[i] = bias_hidden[i]
            bias_hidden[i] += (step_size * delta_hidden[i]) + (alpha * bias_diff)
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


def calculate_rmse():
    # RMSE for training data
    rmse_training_sum = 0
    for k in range(num_rows_training):
        correct_output = float(training_data[k, num_inputs])
        sum_output = 0
        for i in range(num_nodes):
            sum_hidden[i] = 0
            for j in range(num_inputs):
                sum_hidden[i] += float(training_data[k, j]) * weights_input_hidden[i, j]
            sum_hidden[i] += bias_hidden[i]
            u_hidden[i] = 1 / (1 + np.exp(-sum_hidden[i]))
            sum_output += u_hidden[i] * weights_hidden_output[i]
        sum_output += bias_output
        u_output = 1 / (1 + np.exp(-sum_output))
        rmse_training_sum += ((correct_output - u_output) ** 2) / num_rows_training
    rmse_training = np.sqrt(rmse_training_sum)
    rmse_training_file.write(str(float(rmse_training)) + "\n")


def validate():
    # Passing Validation Data every 5 epochs
    if epoch_count % 5 == 0:
        rmse_validation_sum = 0
        if epoch_count != 5:
            rmse_validation_old = rmse_validation
        for k in range(num_rows_validation):
            correct_output = float(validation_data[k, num_inputs])
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
