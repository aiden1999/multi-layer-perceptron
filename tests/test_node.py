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
