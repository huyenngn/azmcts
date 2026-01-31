"""Tests for belief.tree module."""

import pytest

from belief import tree


class TestEdgeStats:
    """Test EdgeStats dataclass."""

    def test_default_values(self) -> None:
        """Test default values for EdgeStats."""
        edge = tree.EdgeStats()
        assert edge.n == 0
        assert edge.w == 0.0
        assert edge.p == 0.0

    def test_q_property_zero_visits(self) -> None:
        """Test q property returns 0 when n is 0."""
        edge = tree.EdgeStats(n=0, w=10.0)
        assert edge.q == 0.0

    def test_q_property_with_visits(self) -> None:
        """Test q property calculates average correctly."""
        edge = tree.EdgeStats(n=5, w=10.0)
        assert edge.q == 2.0

    def test_negative_values(self) -> None:
        """Test q property with negative total value."""
        edge = tree.EdgeStats(n=4, w=-8.0)
        assert edge.q == -2.0


class TestNode:
    """Test Node class."""

    def test_node_initialization(self) -> None:
        """Test node is initialized with correct defaults."""
        node = tree.Node(obs_key="test_obs", player_to_act=0)
        assert node.obs_key == "test_obs"
        assert node.player_to_act == 0
        assert node.is_expanded is False
        assert len(node.edges) == 0
        assert len(node.legal_actions) == 0
        assert node.n == 0

    def test_get_most_visited_action_basic(self) -> None:
        """Test get_most_visited_action returns action with highest visit count."""
        node = tree.Node(obs_key="test", player_to_act=0)
        node.edges[0] = tree.EdgeStats(n=5)
        node.edges[1] = tree.EdgeStats(n=10)
        node.edges[2] = tree.EdgeStats(n=3)

        assert node.get_most_visited_action() == 1

    def test_get_most_visited_action_filtered(self) -> None:
        """Test get_most_visited_action with filtered action list."""
        node = tree.Node(obs_key="test", player_to_act=0)
        node.edges[0] = tree.EdgeStats(n=5)
        node.edges[1] = tree.EdgeStats(n=10)
        node.edges[2] = tree.EdgeStats(n=3)

        # Only consider actions 0 and 2
        assert node.get_most_visited_action(actions=[0, 2]) == 0

    def test_get_most_visited_action_keyerror_bug(self) -> None:
        """Test handling actions not in edges (uses default EdgeStats with n=0).

        This reproduces the scenario from the eval script where an action
        from the legal actions list is not present in node.edges.
        Actions not in edges are treated as having 0 visits.
        """
        node = tree.Node(obs_key="test", player_to_act=0)
        node.edges[1] = tree.EdgeStats(n=10)
        node.edges[2] = tree.EdgeStats(n=3)

        # Action 0 is in the actions list but not in edges
        # Should treat it as n=0 and return 1 (highest visits)
        assert node.get_most_visited_action(actions=[0, 1, 2]) == 1

    def test_get_most_visited_action_all_zero_visits(self) -> None:
        """Test when all actions have zero visits (not in edges)."""
        node = tree.Node(obs_key="test", player_to_act=0)
        node.edges[5] = tree.EdgeStats(n=10)  # Not in the actions list

        # All actions [0,1,2] have 0 visits (not in edges)
        # Should return first one (0) since they're all equal
        result = node.get_most_visited_action(actions=[0, 1, 2])
        assert result in [0, 1, 2]  # Any is valid since all have n=0

    def test_get_most_visited_action_empty_edges(self) -> None:
        """Test get_most_visited_action with no edges."""
        node = tree.Node(obs_key="test", player_to_act=0)

        # Should raise ValueError from max() on empty sequence
        with pytest.raises(ValueError):
            node.get_most_visited_action()


class TestBeliefTree:
    """Test BeliefTree class."""

    def test_initialization(self) -> None:
        """Test belief tree initialization."""
        belief_tree = tree.BeliefTree()
        assert len(belief_tree._nodes) == 0

    def test_get_or_create_new_node(self) -> None:
        """Test creating a new node."""
        belief_tree = tree.BeliefTree()
        node = belief_tree.get_or_create("obs1", player_to_act=0)

        assert node.obs_key == "obs1"
        assert node.player_to_act == 0
        assert len(belief_tree._nodes) == 1
        assert "obs1" in belief_tree._nodes

    def test_get_or_create_existing_node(self) -> None:
        """Test retrieving an existing node."""
        belief_tree = tree.BeliefTree()
        node1 = belief_tree.get_or_create("obs1", player_to_act=0)
        node1.n = 5  # Modify the node

        node2 = belief_tree.get_or_create("obs1", player_to_act=0)
        # Should return the same node
        assert node2 is node1
        assert node2.n == 5
        assert len(belief_tree._nodes) == 1

    def test_multiple_nodes(self) -> None:
        """Test creating multiple nodes."""
        belief_tree = tree.BeliefTree()
        node1 = belief_tree.get_or_create("obs1", player_to_act=0)
        node2 = belief_tree.get_or_create("obs2", player_to_act=1)

        assert len(belief_tree._nodes) == 2
        assert node1 is not node2
        assert belief_tree._nodes["obs1"] is node1
        assert belief_tree._nodes["obs2"] is node2
