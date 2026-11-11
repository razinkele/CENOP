"""DEPONS parity of the ADD_ARTIFICIAL_FOOD floor (phantom-food regression).

DEPONS ``CellData.eatFood`` acts only on a cell whose food is > 0: it eats from it and then
raises it to the 0.01 floor. A cell without food (outside the food patches) is left at
exactly 0. CENOP's eat kernels used to floor every visited cell, so empty cells acquired
0.01 "phantom" food that porpoises then ate. See
docs/superpowers/specs/2026-11-10-phantom-food-investigation.md.
"""

import numpy as np
import pytest

from cenop.optimizations.kernels import eat_food_kernel, eat_food_kernel_v2

MIN_FOOD = 0.01

# Four agents: two on empty cells, one on a food cell, one on a food cell below the floor.
XI = np.array([0, 3, 1, 2], dtype=np.int32)
YI = np.array([0, 3, 1, 2], dtype=np.int32)


def _grid():
    food = np.zeros((4, 4), dtype=np.float32)
    food[1, 1] = 0.5
    food[2, 2] = 0.005
    return food


def _assert_depons_floor(food, eaten):
    empty = _grid() == 0.0
    assert np.all(food[empty] == 0.0), "empty cells must stay at exactly 0 (no phantom food)"
    assert eaten[0] == 0.0 and eaten[1] == 0.0
    # Food cell eaten at frac 0.99: 0.495 eaten, 0.005 left, raised to the floor.
    assert eaten[2] == pytest.approx(0.495, rel=1e-6)
    assert food[1, 1] == pytest.approx(MIN_FOOD)
    # Food cell below the floor, visited by a full (frac 0) porpoise: DEPONS still floors it.
    assert eaten[3] == 0.0
    assert food[2, 2] == pytest.approx(MIN_FOOD)


def test_eat_food_kernel_v2_does_not_create_food_in_empty_cells():
    food = _grid()
    energy = np.array([10.0, 10.0, 10.0, 20.0], dtype=np.float32)
    eaten = np.zeros(4, dtype=np.float32)
    eat_food_kernel_v2(food, XI, YI, energy, eaten, MIN_FOOD, np.zeros_like(food))
    _assert_depons_floor(food, eaten)


def test_eat_food_kernel_v1_does_not_create_food_in_empty_cells():
    food = _grid()
    frac = np.array([0.99, 0.99, 0.99, 0.0], dtype=np.float32)
    eaten = np.zeros(4, dtype=np.float32)
    eat_food_kernel(food, XI, YI, frac, eaten, MIN_FOOD, np.zeros_like(food))
    _assert_depons_floor(food, eaten)


def test_jax_eat_food_does_not_create_food_in_empty_cells():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from cenop.optimizations.jax_kernels import jax_eat_food

    energy = jnp.array([10.0, 10.0, 10.0, 20.0], dtype=jnp.float32)
    active = jnp.ones(4, dtype=bool)
    eaten, food = jax_eat_food(jnp.asarray(_grid()), XI, YI, energy, MIN_FOOD, active)
    _assert_depons_floor(np.asarray(food), np.asarray(eaten))


def test_jax_eat_food_ignores_inactive_agents():
    """Dead slots keep stale positions; they must neither eat nor floor a cell."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from cenop.optimizations.jax_kernels import jax_eat_food

    food0 = _grid()
    energy = jnp.full(4, 10.0, dtype=jnp.float32)
    active = jnp.array([True, True, False, False])
    eaten, food = jax_eat_food(jnp.asarray(food0), XI, YI, energy, MIN_FOOD, active)
    np.testing.assert_array_equal(np.asarray(food), food0)
    np.testing.assert_array_equal(np.asarray(eaten), np.zeros(4, dtype=np.float32))


def _cython_post_crw_on_patchy_grid():
    from cenop.optimizations.tick_cython import cython_depons_post_crw

    rng = np.random.default_rng(7)
    n = 400
    food = np.zeros((200, 200), dtype=np.float32)
    food[80:120, 80:120] = 0.5
    empty = food == 0.0
    out_food = np.zeros(n, dtype=np.float32)
    cython_depons_post_crw(
        rng.uniform(5, 195, n).astype(np.float32),
        rng.uniform(5, 195, n).astype(np.float32),
        rng.uniform(0, 360, n).astype(np.float32),
        rng.normal(0, 10, n).astype(np.float64),
        rng.uniform(0.5, 1.5, n).astype(np.float64),
        rng.uniform(5, 15, n).astype(np.float32),
        np.ones(n, dtype=np.uint8),
        np.zeros(n, dtype=np.uint8),
        np.zeros(n, dtype=np.uint8),
        rng.normal(0, 20, n).astype(np.float64),
        rng.uniform(0.5, 1.5, n).astype(np.float64),
        rng.uniform(0, 1, n).astype(np.float32),
        rng.normal(0, 0.1, n).astype(np.float32),
        rng.normal(0, 0.1, n).astype(np.float32),
        food,
        np.full((200, 200), 20.0, dtype=np.float64),
        out_food,
        np.zeros(n, dtype=np.float32),
        rng.random(n),
        0.001,
        4.0,
        4.5,
        1.4,
        1.0,
        0.4,
        1.0,
        200,
        200,
    )
    return food, empty, out_food


def test_cython_post_crw_does_not_create_food_in_empty_cells():
    try:
        from cenop.optimizations.tick_cython import cython_available  # noqa: F401
    except ImportError:
        pytest.skip("Cython module not built")
    food, empty, out_food = _cython_post_crw_on_patchy_grid()
    assert np.all(food[empty] == 0.0), "empty cells must stay at exactly 0 (no phantom food)"
    assert out_food.max() > 0.0, "porpoises on the food block must still eat"


def test_population_steps_keep_empty_cells_empty():
    """End to end through the default backend on a patchy landscape."""
    from cenop.agents.population import PorpoisePopulation
    from cenop.landscape.cell_data import create_homogeneous_landscape
    from cenop.parameters import SimulationParameters

    landscape = create_homogeneous_landscape(width=60, height=60, depth=20.0, food_prob=0.5)
    empty = np.ones((60, 60), dtype=bool)
    empty[20:40, 20:40] = False
    landscape._food_prob[empty] = 0.0
    landscape._food_value[empty] = 0.0
    pop = PorpoisePopulation(
        count=100, params=SimulationParameters(porpoise_count=100), landscape=landscape
    )
    for _ in range(96):
        pop.step()
    assert np.all(landscape._food_value[empty] == 0.0), "phantom food created in empty cells"
