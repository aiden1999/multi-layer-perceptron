from src.classes.node import HiddenNode, Node
from src.classes.weight import Weight

test_node_left = Node(index=0)
test_node_right = HiddenNode(index=0, num_nodes=5)
test_weight = Weight(test_node_left, test_node_right, node_count=4)


def test_update_old_value_works():
    test_weight.value = 10
    expected_value_old = 10
    test_weight.update_old_value()
    actual_value_old = test_weight.value_old
    assert expected_value_old == actual_value_old


def test_update_new_value_no_alpha_works():
    test_weight.value = 5
    step_size = 0.5
    test_weight.right_node.delta = 4
    u = 3
    expected_value = 11
    test_weight.update_new_value(step_size=step_size, u=u)
    actual_value = test_weight.value
    assert expected_value == actual_value
