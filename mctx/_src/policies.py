# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Search policies."""
import functools
from typing import Optional, Any

import chex
import jax
import jax.numpy as jnp

from mctx._src import action_selection
from mctx._src import base
from mctx._src import qtransforms
from mctx._src import search
from mctx._src import seq_halving
from mctx._src.search import batch_update


def muzero_policy(
    params: base.Params,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    num_simulations: int,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
    dirichlet_fraction: chex.Numeric = 0.25,
    dirichlet_alpha: chex.Numeric = 0.3,
    pb_c_init: chex.Numeric = 1.25,
    pb_c_base: chex.Numeric = 19652,
    temperature: chex.Numeric = 1.0) -> base.PolicyOutput[None]:
  """Runs MuZero search and returns the `PolicyOutput`.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    num_simulations: the number of simulations.
    invalid_actions: a mask with invalid actions. Invalid actions
      have ones, valid actions have zeros in the mask. Shape `[B, num_actions]`.
    max_depth: maximum search tree depth allowed during simulation.
    qtransform: function to obtain completed Q-values for a node.
    dirichlet_fraction: float from 0 to 1 interpolating between using only the
      prior policy or just the Dirichlet noise.
    dirichlet_alpha: concentration parameter to parametrize the Dirichlet
      distribution.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    temperature: temperature for acting proportionally to
      `visit_counts**(1 / temperature)`.

  Returns:
    `PolicyOutput` containing the proposed action, action_weights and the used
    search tree.
  """
  rng_key, dirichlet_rng_key, search_rng_key = jax.random.split(rng_key, 3)

  # Adding Dirichlet noise.
  noisy_logits = _get_logits_from_probs(
      _add_dirichlet_noise(
          dirichlet_rng_key,
          jax.nn.softmax(root.prior_logits),
          dirichlet_fraction=dirichlet_fraction,
          dirichlet_alpha=dirichlet_alpha))
  root = root.replace(
      prior_logits=_mask_invalid_actions(noisy_logits, invalid_actions))

  # Running the search.
  interior_action_selection_fn = functools.partial(
      action_selection.muzero_action_selection,
      pb_c_base=pb_c_base,
      pb_c_init=pb_c_init,
      qtransform=qtransform)
  root_action_selection_fn = functools.partial(
      interior_action_selection_fn,
      depth=0)
  search_tree = search.search(
      params=params,
      rng_key=search_rng_key,
      root=root,
      recurrent_fn=recurrent_fn,
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn,
      num_simulations=num_simulations,
      max_depth=max_depth,
      invalid_actions=invalid_actions)

  # Sampling the proposed action proportionally to the visit counts.
  summary = search_tree.summary()
  action_weights = summary.visit_probs
  action_logits = _apply_temperature(
      _get_logits_from_probs(action_weights), temperature)
  action = jax.random.categorical(rng_key, action_logits)
  return base.PolicyOutput(
      action=action,
      action_weights=action_weights,
      search_tree=search_tree)


def muzero_policy_for_action_sequence(
    params: base.Params,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    num_simulations: int,
    stopping_criteria_fn: base.StoppingCriteriaFn = lambda x: False,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
    dirichlet_fraction: chex.Numeric = 0.25,
    dirichlet_alpha: chex.Numeric = 0.3,
    pb_c_init: chex.Numeric = 1.25,
    pb_c_base: chex.Numeric = 19652,
    temperature: chex.Numeric = 1.0,
    extra_data: Any = None,
    num_actions_to_generate: int = 1,
) -> base.PolicyOutput[None]:
  """Runs MuZero search to generate a sequence of actions and returns the
  `PolicyOutput`. For comparison, `muzero_policy` returns only the prediction
  for one action, not a sequence.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    stopping_criteria_fn: a callable to be called when an action has been
      generated, which takes as args `(embedding)` and returns a `bool`.
      If True is returned, the generation processes stops.
    num_simulations: the number of simulations per generated action
    invalid_actions: a mask with invalid actions. Invalid actions
      have ones, valid actions have zeros in the mask. Shape `[B, num_actions]`.
    max_depth: maximum search tree depth allowed during simulation.
    qtransform: function to obtain completed Q-values for a node.
    dirichlet_fraction: float from 0 to 1 interpolating between using only the
      prior policy or just the Dirichlet noise.
    dirichlet_alpha: concentration parameter to parametrize the Dirichlet
      distribution.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    temperature: temperature for acting proportionally to
      `visit_counts**(1 / temperature)`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    num_actions_to_generate: Number of actions to generate. For each action,
      `num_simulations` simulations are performed and the most visited child
      action is taken. Further simulations are recursively performed
      on the subtree, until all `num_action_to_generate` are generated.

  Returns:
    `PolicyOutput` containing the proposed action, action_weights and the used
    search tree.
  """
  if invalid_actions is not None:
    raise NotImplementedError(
      "`invalid_actions` are not supported as the root position in the tree"
      " changes, but `mctx._src.search.simulate` starts always at depth zero,"
      " taking into account the old invalid_actions of the initial root."
      " To support this, a redesign is needed: 1. Are invalid actions different"
      " for different nodes in the tree, or does only the root have invalid"
      " actions? 2. Where to fetch invalid_actions of non-root nodes during"
      " the search?")

  # TODO `batch_size>1` needs to be verified for correctness.
  # batch_size = root.value.shape[0]
  # if batch_size > 1:
  #   raise NotImplementedError(
  #     "Only `batch_size=1` is supported at the moment. To support different"
  #     " batch sizes, the search.backward function needs to be verified/updated"
  #     " to properly move up the tree and stop at the root when the batches"
  #     " of expanded nodes are at different depths.")

  from mctx import Tree
  from mctx._src.search import simulate, expand, backward, instantiate_tree_from_root

  rng_key, dirichlet_rng_key, search_rng_key = jax.random.split(rng_key, 3)

  # Adding Dirichlet noise.
  noisy_logits = _get_logits_from_probs(
      _add_dirichlet_noise(
          dirichlet_rng_key,
          jax.nn.softmax(root.prior_logits),
          dirichlet_fraction=dirichlet_fraction,
          dirichlet_alpha=dirichlet_alpha))
  root = root.replace(
      prior_logits=_mask_invalid_actions(noisy_logits, invalid_actions))

  # Running the search.
  interior_action_selection_fn = functools.partial(
      action_selection.muzero_action_selection,
      pb_c_base=pb_c_base,
      pb_c_init=pb_c_init,
      qtransform=qtransform)
  root_action_selection_fn = functools.partial(
      interior_action_selection_fn,
      depth=0)

  # ~~~ mctx.search ~~~ #
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  batch_range = jnp.arange(batch_size)
  if max_depth is None:
    max_depth = num_simulations * num_actions_to_generate
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  def generate_next_action_inner(sim, loop_state):
    rng_key, tree, max_depth = loop_state
    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
    # simulate is vmapped and expects batched rng keys.
    simulate_keys = jax.random.split(simulate_key, batch_size)
    parent_index, action = simulate(
        simulate_keys, tree, action_selection_fn, max_depth)
    # A node first expanded on simulation `i`, will have node index `i`.
    # Node 0 does not need to correspond to the root node,
    # because the root changes as actions get performed.
    next_node_index = tree.children_index[batch_range, parent_index, action]
    next_node_index = jnp.where(next_node_index == Tree.UNVISITED,
                                sim + 1, next_node_index)
    tree = expand(
        params, expand_key, tree, recurrent_fn, parent_index,
        action, next_node_index)
    tree = backward(tree, next_node_index)
    loop_state = rng_key, tree, max_depth
    return loop_state

  def generate_next_action(generated_action, loop_state):
    rng_key, tree, max_depth, performed_actions, stop_generating = loop_state

    # Peform simulations and upate the tree.
    _, tree, _ = jax.lax.fori_loop(
      generated_action * num_simulations,
      (generated_action + 1) * num_simulations,
      generate_next_action_inner, (rng_key, tree, max_depth))

    # Sampling the action to be performed proportionally to the visit counts.
    summary = tree.summary()
    action_weights = summary.visit_probs
    action_logits = _apply_temperature(
        _get_logits_from_probs(action_weights), temperature)
    action = jax.random.categorical(rng_key, action_logits)

    # Make the new action and update the root.
    # TODO refactor/simplify the updating of the root index
    def update_new_root_index(batch_idx, new_root_index):
      return new_root_index.at[batch_idx].set(
        tree.children_index[batch_idx, tree.root_index[batch_idx], action[batch_idx]])
    new_root_index = jax.lax.fori_loop(0, batch_size, update_new_root_index,
                                       jnp.full_like(tree.root_index, -1))

    # TODO Will qtransforms know how to deal with the updated root?
    tree = tree.replace(
      root_index=tree.root_index.at[:].set(new_root_index),
      # Parents of the root are set to Tree.NO_PARENT
      parents=batch_update(tree.parents, jnp.full((batch_size,), Tree.NO_PARENT), new_root_index),
    )

    performed_actions = performed_actions.at[:, generated_action].set(action)

    # Call `stopping_criteria_fn` to determine if generation should be stopped.
    root_embedding = jax.tree_map(
      lambda x: x[batch_range, tree.root_index], tree.embeddings)
    stop_generating = stopping_criteria_fn(root_embedding)

    loop_state = rng_key, tree, max_depth - 1, performed_actions, stop_generating
    return loop_state

  def generate_next_action_stopping_wrapper(generated_action, loop_state):
    _, _, max_depth, _, stop_generating = loop_state

    # TODO is there a nicer way to crate the simple condition `stop_generating or max_depth==0`?
    stop_generating_is_true_or_max_depth_is_zero = jax.lax.cond(stop_generating, lambda: True, lambda: max_depth == 0)

    return jax.lax.cond(stop_generating_is_true_or_max_depth_is_zero,
                        lambda _, x: x, generate_next_action, generated_action, loop_state)

  # Allocate all necessary storage.
  tree = instantiate_tree_from_root(
    root, num_actions_to_generate * num_simulations,
    root_invalid_actions=invalid_actions, extra_data=extra_data)

  performed_actions = jnp.full((batch_size, num_actions_to_generate), -1, dtype=jnp.int32)
  _, tree, _, performed_actions, _ = jax.lax.fori_loop(
    lower=0,
    upper=num_actions_to_generate,
    body_fun=generate_next_action_stopping_wrapper,
    init_val=(rng_key, tree, max_depth, performed_actions, False),
  )

  return base.PolicyOutput(
      action=performed_actions,
      action_weights=jnp.empty((0,)),
      search_tree=tree)

def gumbel_muzero_policy(
    params: base.Params,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    num_simulations: int,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_completed_by_mix_value,
    max_num_considered_actions: int = 16,
    gumbel_scale: chex.Numeric = 1.,
) -> base.PolicyOutput[action_selection.GumbelMuZeroExtraData]:
  """Runs Gumbel MuZero search and returns the `PolicyOutput`.

  This policy implements Full Gumbel MuZero from
  "Policy improvement by planning with Gumbel".
  https://openreview.net/forum?id=bERaNdoegnO

  At the root of the search tree, actions are selected by Sequential Halving
  with Gumbel. At non-root nodes (aka interior nodes), actions are selected by
  the Full Gumbel MuZero deterministic action selection.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    num_simulations: the number of simulations.
    invalid_actions: a mask with invalid actions. Invalid actions
      have ones, valid actions have zeros in the mask. Shape `[B, num_actions]`.
    max_depth: maximum search tree depth allowed during simulation.
    qtransform: function to obtain completed Q-values for a node.
    max_num_considered_actions: the maximum number of actions expanded at the
      root node. A smaller number of actions will be expanded if the number of
      valid actions is smaller.
    gumbel_scale: scale for the Gumbel noise. Evalution on perfect-information
      games can use gumbel_scale=0.0.

  Returns:
    `PolicyOutput` containing the proposed action, action_weights and the used
    search tree.
  """
  # Masking invalid actions.
  root = root.replace(
      prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions))

  # Generating Gumbel.
  rng_key, gumbel_rng = jax.random.split(rng_key)
  gumbel = gumbel_scale * jax.random.gumbel(
      gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)

  # Searching.
  extra_data = action_selection.GumbelMuZeroExtraData(root_gumbel=gumbel)
  search_tree = search.search(
      params=params,
      rng_key=rng_key,
      root=root,
      recurrent_fn=recurrent_fn,
      root_action_selection_fn=functools.partial(
          action_selection.gumbel_muzero_root_action_selection,
          num_simulations=num_simulations,
          max_num_considered_actions=max_num_considered_actions,
          qtransform=qtransform,
      ),
      interior_action_selection_fn=functools.partial(
          action_selection.gumbel_muzero_interior_action_selection,
          qtransform=qtransform,
      ),
      num_simulations=num_simulations,
      max_depth=max_depth,
      invalid_actions=invalid_actions,
      extra_data=extra_data)
  summary = search_tree.summary()

  # Acting with the best action from the most visited actions.
  # The "best" action has the highest `gumbel + logits + q`.
  # Inside the minibatch, the considered_visit can be different on states with
  # a smaller number of valid actions.
  considered_visit = jnp.max(summary.visit_counts, axis=-1, keepdims=True)
  # The completed_qvalues include imputed values for unvisited actions.
  completed_qvalues = jax.vmap(qtransform, in_axes=[0, None])(
      search_tree, search_tree.INITIAL_ROOT_INDEX)
  to_argmax = seq_halving.score_considered(
      considered_visit, gumbel, root.prior_logits, completed_qvalues,
      summary.visit_counts)
  action = action_selection.masked_argmax(to_argmax, invalid_actions)

  # Producing action_weights usable to train the policy network.
  completed_search_logits = _mask_invalid_actions(
      root.prior_logits + completed_qvalues, invalid_actions)
  action_weights = jax.nn.softmax(completed_search_logits)
  return base.PolicyOutput(
      action=action,
      action_weights=action_weights,
      search_tree=search_tree)


def _mask_invalid_actions(logits, invalid_actions):
  """Returns logits with zero mass to invalid actions."""
  if invalid_actions is None:
    return logits
  chex.assert_equal_shape([logits, invalid_actions])
  logits = logits - jnp.max(logits, axis=-1, keepdims=True)
  # At the end of an episode, all actions can be invalid. A softmax would then
  # produce NaNs, if using -inf for the logits. We avoid the NaNs by using
  # a finite `min_logit` for the invalid actions.
  min_logit = jnp.finfo(logits.dtype).min
  return jnp.where(invalid_actions, min_logit, logits)


def _get_logits_from_probs(probs):
  tiny = jnp.finfo(probs).tiny
  return jnp.log(jnp.maximum(probs, tiny))


def _add_dirichlet_noise(rng_key, probs, *, dirichlet_alpha,
                         dirichlet_fraction):
  """Mixes the probs with Dirichlet noise."""
  chex.assert_rank(probs, 2)
  chex.assert_type([dirichlet_alpha, dirichlet_fraction], float)

  batch_size, num_actions = probs.shape
  noise = jax.random.dirichlet(
      rng_key,
      alpha=jnp.full([num_actions], fill_value=dirichlet_alpha),
      shape=(batch_size,))
  noisy_probs = (1 - dirichlet_fraction) * probs + dirichlet_fraction * noise
  return noisy_probs


def _apply_temperature(logits, temperature):
  """Returns `logits / temperature`, supporting also temperature=0."""
  # The max subtraction prevents +inf after dividing by a small temperature.
  logits = logits - jnp.max(logits, keepdims=True, axis=-1)
  tiny = jnp.finfo(logits.dtype).tiny
  return logits / jnp.maximum(tiny, temperature)
