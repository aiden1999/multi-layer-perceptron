from src.classes.node import Node


def test_reset_sum_works():
    test_node = Node(index=0)
    test_node.sum = 26
    expected_sum = 0
    test_node.reset_sum()
    actual_sum = test_node.sum
    assert expected_sum == actual_sum
