from src.classes.node import HiddenNode, Node, OutputNode


def test_reset_sum_works():
    test_node = Node(index=0)
    test_node.sum = 26
    expected_sum = 0
    test_node.reset_sum()
    actual_sum = test_node.sum
    assert expected_sum == actual_sum


def test_activation_function_works():
    test_node = Node(index=0)
    test_node.sum = 0
    expected_u = 0.5
    test_node.activation_function()
    actual_u = test_node.u
    assert expected_u == actual_u


def test_calculate_f_prime_works():
    test_node = Node(index=0)
    test_node.u = 5
    expected_f_prime = -20
    test_node.calculate_f_prime()
    actual_f_prime = test_node.f_prime
    assert expected_f_prime == actual_f_prime


def test_update_old_bias_works():
    test_node = Node(index=0)
    test_node.bias = 3
    expected_old_bias = 3
    test_node.update_old_bias()
    actual_old_bias = test_node.bias_old
    assert expected_old_bias == actual_old_bias


def test_update_new_bias_no_alpha_works():
    test_node = Node(index=0)
    test_node.bias = 10
    step_size = 0.5
    test_node.delta = 4
    expected_bias = 12
    test_node.update_new_bias(step_size)
    actual_bias = test_node.bias
    assert expected_bias == actual_bias


def test_update_new_bias_with_alpha_works():
    test_node = Node(index=0)
    test_node.bias = 10
    step_size = 0.5
    test_node.delta = 4
    alpha = 0.9
    test_node.bias_diff = 10
    expected_bias = 21
    test_node.update_new_bias(step_size=step_size, alpha=alpha)
    actual_bias = test_node.bias
    assert expected_bias == actual_bias


def test_update_bias_diff_works():
    test_node = Node(index=0)
    test_node.bias = 10
    test_node.bias_old = 6
    expected_bias_diff = 4
    test_node.update_bias_diff()
    actual_bias_diff = test_node.bias_diff
    assert expected_bias_diff == actual_bias_diff


def test_reset_bias_for_bias():
    test_node = Node(index=0)
    test_node.bias = 2
    test_node.bias_old = 1
    expected_bias = 1
    test_node.reset_bias()
    actual_bias = test_node.bias
    assert expected_bias == actual_bias


def test_reset_bias_for_bias_old():
    test_node = Node(index=0)
    test_node.bias = 2
    test_node.bias_old = 1
    expected_bias_old = 0
    test_node.reset_bias()
    actual_bias_old = test_node.bias_old
    assert expected_bias_old == actual_bias_old


def test_hidden_node_calculate_delta_works():
    test_hidden_node = HiddenNode(index=0, num_nodes=5)
    weight = 2
    delta_output = 3
    test_hidden_node.f_prime = 4
    expected_delta = 24
    test_hidden_node.calculate_delta(weight=weight, delta_output=delta_output)
    actual_delta = test_hidden_node.delta
    assert expected_delta == actual_delta


def test_output_node_calculate_delta_works():
    test_output_node = OutputNode(index=0, num_nodes=5)
    correct_output = 5
    test_output_node.u = 2
    test_output_node.f_prime = 3
    expected_delta = 9
    test_output_node.calculate_delta(correct_output=correct_output)
    actual_delta = test_output_node.delta
    assert expected_delta == actual_delta
