"""A unit that verifies that muzero_policy_for_action_sequence
 compiles for different number for steps and batch sizes."""
import functools

import jax
from absl import logging
from absl.testing import parameterized

import mctx
from mctx._src.tests.tree_test import _prepare_root, _prepare_recurrent_fn


## Uncomment for debugging in order to be able to inspect values in the debugger:
# from jax.config import config;
# config.update('jax_disable_jit', True)

class MuzeroForActionSequenceTest(parameterized.TestCase):
  @parameterized.named_parameters(
    ("qtransform_by_min_max", "qtransform_by_min_max"),
    ("qtransform_by_parent_and_siblings", "qtransform_by_parent_and_siblings"),
    ("qtransform_noop", "qtransform_noop"),
  )
  def test_1_step(self, qtransform):
    policy_output = self._run(1, 10, 1, qtransform, f"/tmp/muzero-for-action-sequence-1x10-{qtransform}.png")
    print(policy_output)

  @parameterized.named_parameters(
    ("qtransform_by_min_max", "qtransform_by_min_max"),
    ("qtransform_by_parent_and_siblings", "qtransform_by_parent_and_siblings"),
    ("qtransform_noop", "qtransform_noop"),
  )
  def test_2_steps(self, qtransform):
    policy_output = self._run(1, 10, 2, qtransform, f"/tmp/muzero-for-action-sequence-2x10-{qtransform}.png")
    print(policy_output)

  @parameterized.named_parameters(
    ("qtransform_by_min_max", "qtransform_by_min_max"),
    ("qtransform_by_parent_and_siblings", "qtransform_by_parent_and_siblings"),
    ("qtransform_noop", "qtransform_noop"),
  )
  def test_3_steps(self, qtransform):
    policy_output = self._run(1, 10, 3, qtransform, f"/tmp/muzero-for-action-sequence-3x10-{qtransform}.png")
    print(policy_output)

  @parameterized.named_parameters(
    ("qtransform_by_min_max", "qtransform_by_min_max"),
    ("qtransform_by_parent_and_siblings", "qtransform_by_parent_and_siblings"),
    ("qtransform_noop", "qtransform_noop"),
  )
  def test_4_steps(self, qtransform):
    policy_output = self._run(1, 10, 4, qtransform, f"/tmp/muzero-for-action-sequence-3x10-{qtransform}.png")
    print(policy_output)

  @parameterized.named_parameters(
    ("qtransform_by_min_max", "qtransform_by_min_max"),
    ("qtransform_by_parent_and_siblings", "qtransform_by_parent_and_siblings"),
    ("qtransform_noop", "qtransform_noop"),
  )
  def test_5_steps(self, qtransform):
    policy_output = self._run(1, 10, 5, qtransform, f"/tmp/muzero-for-action-sequence-5x10-{qtransform}.png")
    print(policy_output)

  @parameterized.named_parameters(
    ("qtransform_by_min_max", "qtransform_by_min_max"),
    ("qtransform_by_parent_and_siblings", "qtransform_by_parent_and_siblings"),
  )
  def test_batch_1_step(self, qtransform):
    policy_output = self._run(3, 10, 1, qtransform, f"/tmp/muzero-for-action-sequence-bs3-1x10-{qtransform}.png")
    print(policy_output)

  @parameterized.named_parameters(
    ("qtransform_by_min_max", "qtransform_by_min_max"),
    ("qtransform_by_parent_and_siblings", "qtransform_by_parent_and_siblings"),
    ("qtransform_noop", "qtransform_noop"),
  )
  def test_batch_3_steps(self, qtransform):
    policy_output = self._run(3, 50, 3, qtransform, f"/tmp/muzero-for-action-sequence-bs3-3x50-{qtransform}.png")
    print(policy_output)

  def _run(self, batch_size, num_simulations, num_actions_to_generate,
           qtransform, draw_graph_path=None):
    num_actions = 82

    algorithm_config = {
      "dirichlet_alpha": 0.3,
      "dirichlet_fraction": 0.0,
      "pb_c_base": 19652,
      "pb_c_init": 1.25,
    }

    if qtransform == "qtransform_by_min_max":
      env_config = {
        "discount": -1.0,
        "zero_reward": True
      }
      qtransform_kwargs = {
        "max_value": 1.0,
        "min_value": -1.0
      }
    elif qtransform == "qtransform_by_parent_and_siblings":
      env_config = {
        "discount": 0.997,
        "zero_reward": False
      }
      qtransform_kwargs = {}
    elif qtransform == "qtransform_noop":
      env_config = {
        "discount": 1,
        "zero_reward": False
      }
      qtransform_kwargs = {}
    else:
      raise ValueError()

    qtransform = functools.partial(getattr(mctx, qtransform), **qtransform_kwargs)

    def run_policy():
      return mctx.muzero_policy_for_action_sequence(
        num_actions_to_generate=num_actions_to_generate,
        num_simulations=num_simulations,
        # return mctx.muzero_policy(
        #   num_simulations=num_simulations * num_actions_to_generate,
        params=(),
        rng_key=jax.random.PRNGKey(1),
        root=_prepare_root(batch_size=batch_size, num_actions=num_actions),
        recurrent_fn=_prepare_recurrent_fn(num_actions, **env_config),
        qtransform=qtransform,
        **algorithm_config,
      )

    policy_output = jax.jit(run_policy)()
    logging.info("Done search.")

    if draw_graph_path is not None:
      from examples.visualization_demo import convert_tree_to_graph
      graph = convert_tree_to_graph(policy_output.search_tree)
      graph.draw(draw_graph_path, prog="dot")

    return policy_output
