from src.classes.node import Node


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
