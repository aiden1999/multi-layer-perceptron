def test():
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
