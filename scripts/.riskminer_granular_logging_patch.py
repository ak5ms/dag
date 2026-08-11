from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        if new in text:
            return text
        raise SystemExit(f"missing patch target: {label}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# MCTS internal tracing
# ---------------------------------------------------------------------------
p = Path('src/flows/riskminer/mcts.py')
s = p.read_text()
s = replace_once(
    s,
    'from collections.abc import Mapping, Sequence\n',
    'from collections.abc import Callable, Mapping, Sequence\n',
    'mcts Callable import',
)
s = replace_once(
    s,
    '    nodes: dict[tuple, TreeNode]\n\n    def _node(self, state: RPNState) -> TreeNode:\n',
    '''    nodes: dict[tuple, TreeNode]\n\n    def _emit(self, event: str, **payload: object) -> None:\n        sink = getattr(self, "on_event", None)\n        if sink is not None:\n            sink(event, dict(payload))\n\n    def _edge_snapshot(self, node: TreeNode) -> list[dict[str, object]]:\n        total = max(1, node.visits + node.virtual_visits)\n        rows: list[dict[str, object]] = []\n        for token_id, edge in node.edges.items():\n            visits = edge.visits + edge.virtual_visits\n            bonus = (\n                self.config.exploration\n                * edge.prior\n                * math.sqrt(total)\n                / (1 + visits)\n            )\n            rows.append({\n                "token": self.environment.vocabulary.by_id[token_id].name,\n                "prior": float(edge.prior),\n                "q": float(edge.q),\n                "visits": int(edge.visits),\n                "virtual_visits": int(edge.virtual_visits),\n                "puct": float(edge.q + bonus),\n            })\n        return sorted(rows, key=lambda row: (-float(row["puct"]), str(row["token"])))\n\n    def _node(self, state: RPNState) -> TreeNode:\n''',
    'mcts emit helpers',
)

s = replace_once(
    s,
    '''        if unvisited:\n            token_id, edge = max(\n                unvisited, key=lambda item: (item[1].prior, -item[0])\n            )\n            return token_id, edge, True\n''',
    '''        if unvisited:\n            token_id, edge = max(\n                unvisited, key=lambda item: (item[1].prior, -item[0])\n            )\n            self._emit(\n                "mcts_node_choice",\n                state_rpn=self._render_token_ids(state.token_ids),\n                stack_size=len(state.stack),\n                node_visits=node.visits,\n                node_virtual_visits=node.virtual_visits,\n                legal_count=len(legal),\n                exposed_count=len(node.edges),\n                allowed_count=allowed_count,\n                reason="new_edge",\n                selected=self.environment.vocabulary.by_id[token_id].name,\n                selected_prior=float(edge.prior),\n                selected_q=float(edge.q),\n                edges=self._edge_snapshot(node),\n            )\n            return token_id, edge, True\n''',
    'mcts unvisited choice logging',
)
s = replace_once(
    s,
    '''        token_id, edge = max(node.edges.items(), key=puct)\n        return token_id, edge, False\n''',
    '''        token_id, edge = max(node.edges.items(), key=puct)\n        self._emit(\n            "mcts_node_choice",\n            state_rpn=self._render_token_ids(state.token_ids),\n            stack_size=len(state.stack),\n            node_visits=node.visits,\n            node_virtual_visits=node.virtual_visits,\n            legal_count=len(legal),\n            exposed_count=len(node.edges),\n            allowed_count=allowed_count,\n            reason="puct",\n            selected=self.environment.vocabulary.by_id[token_id].name,\n            selected_prior=float(edge.prior),\n            selected_q=float(edge.q),\n            edges=self._edge_snapshot(node),\n        )\n        return token_id, edge, False\n''',
    'mcts puct choice logging',
)

# Rollout choice logging, including END reasons and class-balanced weights.
s = replace_once(
    s,
    '''        if (\n            end_id in legal\n            and rng.random() < self.config.rollout_end_probability\n        ):\n            return end_id\n        choices = [token_id for token_id in legal if token_id != end_id]\n        if not choices:\n            return end_id\n''',
    '''        if (\n            end_id in legal\n            and rng.random() < self.config.rollout_end_probability\n        ):\n            self._emit(\n                "mcts_rollout_choice",\n                state_rpn=self._render_token_ids(state.token_ids),\n                stack_size=len(state.stack),\n                legal_count=len(legal),\n                selected=self.environment.vocabulary.by_id[end_id].name,\n                reason="end_probability",\n            )\n            return end_id\n        choices = [token_id for token_id in legal if token_id != end_id]\n        if not choices:\n            self._emit(\n                "mcts_rollout_choice",\n                state_rpn=self._render_token_ids(state.token_ids),\n                stack_size=len(state.stack),\n                legal_count=len(legal),\n                selected=self.environment.vocabulary.by_id[end_id].name,\n                reason="only_end",\n            )\n            return end_id\n''',
    'rollout end logging',
)
s = replace_once(
    s,
    '''        weights = [weights_by_id[token_id] for token_id in choices]\n        if not any(weights):\n            weights = [1.0] * len(choices)\n        return rng.choices(choices, weights=weights, k=1)[0]\n''',
    '''        weights = [weights_by_id[token_id] for token_id in choices]\n        if not any(weights):\n            weights = [1.0] * len(choices)\n        selected = rng.choices(choices, weights=weights, k=1)[0]\n        self._emit(\n            "mcts_rollout_choice",\n            state_rpn=self._render_token_ids(state.token_ids),\n            stack_size=len(state.stack),\n            legal_count=len(legal),\n            selected=self.environment.vocabulary.by_id[selected].name,\n            reason="sampled",\n            raw_prior=float(priors.get(selected, 0.0)),\n            adjusted_weight=float(weights_by_id.get(selected, 0.0)),\n            group_sizes={name: len(token_ids) for name, token_ids in groups.items()},\n        )\n        return selected\n''',
    'rollout sampled logging',
)

# Add on_event to standalone RiskMCTS too, for consistency.
s = replace_once(
    s,
    '''        config: RiskMinerConfig | None = None,\n        policy: ActionPolicy | None = None,\n    ) -> None:\n        self.environment = environment\n        self.evaluator = evaluator\n''',
    '''        config: RiskMinerConfig | None = None,\n        policy: ActionPolicy | None = None,\n        on_event: Callable[[str, dict[str, object]], None] | None = None,\n    ) -> None:\n        self.environment = environment\n        self.evaluator = evaluator\n''',
    'RiskMCTS on_event signature',
)
s = replace_once(
    s,
    '''        self.nodes: dict[tuple, TreeNode] = {}\n        self.archive = FormulaArchive(self.config.archive_size)\n\n    def search(self) -> RiskMinerSearchResult:\n''',
    '''        self.nodes: dict[tuple, TreeNode] = {}\n        self.archive = FormulaArchive(self.config.archive_size)\n        self.on_event = on_event\n\n    def search(self) -> RiskMinerSearchResult:\n''',
    'RiskMCTS on_event assignment',
)

# RewardDense constructor gets event sink.
s = replace_once(
    s,
    '''        config: RiskMinerConfig | None = None,\n        policy: ActionPolicy | None = None,\n    ) -> None:\n        self.environment = environment\n        self.reward_model = reward_model\n''',
    '''        config: RiskMinerConfig | None = None,\n        policy: ActionPolicy | None = None,\n        on_event: Callable[[str, dict[str, object]], None] | None = None,\n    ) -> None:\n        self.environment = environment\n        self.reward_model = reward_model\n''',
    'RewardDense on_event signature',
)
s = replace_once(
    s,
    '''        self.nodes: dict[tuple, TreeNode] = {}\n        self.archive = FormulaArchive(self.config.archive_size)\n\n    def search(self) -> RewardDenseSearchResult:\n''',
    '''        self.nodes: dict[tuple, TreeNode] = {}\n        self.archive = FormulaArchive(self.config.archive_size)\n        self.on_event = on_event\n\n    def search(self) -> RewardDenseSearchResult:\n''',
    'RewardDense on_event assignment',
)

# Search lifecycle and simulation-level logging.
s = replace_once(
    s,
    '''        trajectories: list[PolicyTrajectory] = []\n        simulations = rollouts = invalid = pool_updates = 0\n        intermediate_requests = finite_scores = 0\n        while simulations < self.config.simulations:\n            selection = self._select_and_expand(root)\n            for _ in range(self.config.rollouts_per_expansion):\n''',
    '''        trajectories: list[PolicyTrajectory] = []\n        simulations = rollouts = invalid = pool_updates = 0\n        intermediate_requests = finite_scores = 0\n        self._emit(\n            "mcts_search_start",\n            simulations=self.config.simulations,\n            rollouts_per_expansion=self.config.rollouts_per_expansion,\n            max_depth=self.config.max_depth,\n            min_formula_depth=self.config.min_formula_depth,\n            max_tokens=self.config.max_tokens,\n            exploration=self.config.exploration,\n        )\n        while simulations < self.config.simulations:\n            simulation_number = simulations + 1\n            self._emit("mcts_simulation_start", simulation=simulation_number)\n            selection = self._select_and_expand(root)\n            self._emit(\n                "mcts_selection_done",\n                simulation=simulation_number,\n                path=[\n                    self.environment.vocabulary.by_id[token_id].name\n                    for _, token_id in selection.path\n                ],\n                leaf_rpn=self._render_token_ids(selection.leaf.token_ids),\n                leaf_stack_size=len(selection.leaf.stack),\n                leaf_terminated=selection.leaf.terminated,\n            )\n            for rollout_number in range(1, self.config.rollouts_per_expansion + 1):\n                self._emit(\n                    "mcts_rollout_start",\n                    simulation=simulation_number,\n                    rollout=rollout_number,\n                    leaf_rpn=self._render_token_ids(selection.leaf.token_ids),\n                )\n''',
    'RewardDense search start logging',
)
s = replace_once(
    s,
    '''                self._backpropagate_dense(\n                    selection.path, path_returns, path_rewards\n                )\n            simulations += 1\n        return RewardDenseSearchResult(\n            self.archive.entries,\n            tuple(trajectories),\n            SearchMetrics(\n                simulations=simulations,\n                rollouts=rollouts,\n                unique_formula_requests=intermediate_requests,\n                finite_formula_scores=finite_scores,\n                invalid_rollouts=invalid,\n                tree_nodes=len(self.nodes),\n                wall_seconds=time.perf_counter() - started,\n                trajectories=len(trajectories),\n                pool_updates=pool_updates,\n                intermediate_formula_requests=intermediate_requests,\n            ),\n        )\n''',
    '''                self._backpropagate_dense(\n                    selection.path, path_returns, path_rewards\n                )\n            simulations += 1\n            self._emit(\n                "mcts_simulation_done",\n                simulation=simulation_number,\n                trajectories=len(trajectories),\n                invalid_rollouts=invalid,\n                pool_updates=pool_updates,\n                tree_nodes=len(self.nodes),\n                archive_size=len(self.archive.entries),\n            )\n        result = RewardDenseSearchResult(\n            self.archive.entries,\n            tuple(trajectories),\n            SearchMetrics(\n                simulations=simulations,\n                rollouts=rollouts,\n                unique_formula_requests=intermediate_requests,\n                finite_formula_scores=finite_scores,\n                invalid_rollouts=invalid,\n                tree_nodes=len(self.nodes),\n                wall_seconds=time.perf_counter() - started,\n                trajectories=len(trajectories),\n                pool_updates=pool_updates,\n                intermediate_formula_requests=intermediate_requests,\n            ),\n        )\n        self._emit(\n            "mcts_search_done",\n            simulations=simulations,\n            rollouts=rollouts,\n            trajectories=len(trajectories),\n            invalid_rollouts=invalid,\n            pool_updates=pool_updates,\n            tree_nodes=len(self.nodes),\n            archive_size=len(result.archive),\n            best_archive_score=(result.archive[0].score if result.archive else None),\n            wall_seconds=result.metrics.wall_seconds,\n        )\n        return result\n''',
    'RewardDense search done logging',
)

# Selection-edge details after applying selected action.
s = replace_once(
    s,
    '''            child = self.environment.apply(state, token_id)\n            edge.child_key = self.environment.state_key(child)\n            edge.virtual_visits += 1\n''',
    '''            child = self.environment.apply(state, token_id)\n            self._emit(\n                "mcts_selection_edge",\n                step=len(path) + 1,\n                state_rpn=self._render_token_ids(state.token_ids),\n                selected=self.environment.vocabulary.by_id[token_id].name,\n                child_rpn=self._render_token_ids(child.token_ids),\n                expanded=expanded,\n                selected_prior=float(edge.prior),\n                selected_q=float(edge.q),\n                selected_visits=edge.visits,\n                legal_actions=[\n                    self.environment.vocabulary.by_id[action].name for action in legal\n                ],\n            )\n            edge.child_key = self.environment.state_key(child)\n            edge.virtual_visits += 1\n''',
    'RewardDense selection edge logging',
)

# The first occurrence above is RiskMCTS. Patch the remaining RewardDense occurrence too.
if s.count('"mcts_selection_edge"') < 2:
    s = replace_once(
        s,
        '''            child = self.environment.apply(state, token_id)\n            edge.child_key = self.environment.state_key(child)\n            edge.virtual_visits += 1\n''',
        '''            child = self.environment.apply(state, token_id)\n            self._emit(\n                "mcts_selection_edge",\n                step=len(path) + 1,\n                state_rpn=self._render_token_ids(state.token_ids),\n                selected=self.environment.vocabulary.by_id[token_id].name,\n                child_rpn=self._render_token_ids(child.token_ids),\n                expanded=expanded,\n                selected_prior=float(edge.prior),\n                selected_q=float(edge.q),\n                selected_visits=edge.visits,\n                legal_actions=[\n                    self.environment.vocabulary.by_id[action].name for action in legal\n                ],\n            )\n            edge.child_key = self.environment.state_key(child)\n            edge.virtual_visits += 1\n''',
        'second selection edge logging',
    )

# Rollout step and completion logging.
s = replace_once(
    s,
    '''            token_id = self._sample_rollout_action(state, legal, self.rng)\n            child = self.environment.apply(state, token_id)\n            states.append(state)\n''',
    '''            token_id = self._sample_rollout_action(state, legal, self.rng)\n            child = self.environment.apply(state, token_id)\n            self._emit(\n                "mcts_rollout_step",\n                step=len(actions) + 1,\n                state_rpn=self._render_token_ids(state.token_ids),\n                selected=self.environment.vocabulary.by_id[token_id].name,\n                child_rpn=self._render_token_ids(child.token_ids),\n                legal_count=len(legal),\n                child_stack_size=len(child.stack),\n                child_terminated=child.terminated,\n            )\n            states.append(state)\n''',
    'rollout step logging',
)
s = replace_once(
    s,
    '''        return tuple(states), tuple(legal_history), tuple(actions), tuple(resulting)\n\n    def _complete_episode(\n''',
    '''        self._emit(\n            "mcts_rollout_done",\n            actions=[self.environment.vocabulary.by_id[action].name for action in actions],\n            final_rpn=(self._render_token_ids(resulting[-1].token_ids) if resulting else self._render_token_ids(leaf.token_ids)),\n            terminated=bool(resulting and resulting[-1].terminated),\n            steps=len(actions),\n        )\n        return tuple(states), tuple(legal_history), tuple(actions), tuple(resulting)\n\n    def _complete_episode(\n''',
    'rollout done logging',
)

# Invalid episode logging.
s = replace_once(
    s,
    '''            trajectory = PolicyTrajectory(\n                states=tuple(state.token_ids for state in states),\n''',
    '''            self._emit(\n                "mcts_episode_invalid",\n                full_rpn=self._render_token_ids(actions),\n                action_count=len(actions),\n                invalid_reward=self.config.invalid_reward,\n                reason="did_not_reach_END",\n            )\n            trajectory = PolicyTrajectory(\n                states=tuple(state.token_ids for state in states),\n''',
    'invalid episode logging',
)

# Candidate evaluation records and scores.
s = replace_once(
    s,
    '''        intermediate = (\n            dict(self.reward_model.intermediate_rewards(tuple(observations.values())))\n            if observations else {}\n        )\n        finite_count = sum(math.isfinite(value) for value in intermediate.values())\n''',
    '''        candidate_records = [\n            {\n                "canonical_key": repr(key),\n                "depth": value.depth,\n                "expr": repr(value.expr),\n            }\n            for key, value in observations.items()\n        ]\n        self._emit(\n            "mcts_candidates_evaluate",\n            candidate_count=len(candidate_records),\n            candidates=candidate_records,\n        )\n        intermediate = (\n            dict(self.reward_model.intermediate_rewards(tuple(observations.values())))\n            if observations else {}\n        )\n        self._emit(\n            "mcts_candidates_scored",\n            candidate_count=len(candidate_records),\n            candidates=[\n                {\n                    **record,\n                    "score": float(intermediate.get(key, float("nan"))),\n                }\n                for record, (key, _) in zip(candidate_records, observations.items())\n            ],\n        )\n        finite_count = sum(math.isfinite(value) for value in intermediate.values())\n''',
    'candidate evaluation logging',
)

# Archive updates for intermediate candidates.
s = replace_once(
    s,
    '''            self.archive.add(\n                ArchiveEntry(\n                    value.expr, reward, value.depth, value.canonical_key,\n                    self._render_token_ids(state.token_ids),\n                )\n            )\n\n        terminal_parent = states[-1]\n''',
    '''            candidate_rpn = self._render_token_ids(state.token_ids)\n            self.archive.add(\n                ArchiveEntry(\n                    value.expr, reward, value.depth, value.canonical_key,\n                    candidate_rpn,\n                )\n            )\n            self._emit(\n                "mcts_archive_update",\n                rpn=candidate_rpn,\n                depth=value.depth,\n                score=reward,\n                archive_size=len(self.archive.entries),\n                best_score=(self.archive.entries[0].score if self.archive.entries else None),\n            )\n\n        terminal_parent = states[-1]\n''',
    'archive update logging',
)

# Terminal/pool trial logging and result.
s = replace_once(
    s,
    '''        terminal = self.reward_model.terminal_reward(\n            terminal_value, rpn=rpn, individual_score=individual_score\n        )\n        terminal_reward = float(terminal.reward)\n''',
    '''        self._emit(\n            "mcts_terminal_evaluate",\n            rpn=rpn,\n            expr=repr(terminal_value.expr),\n            depth=terminal_value.depth,\n            individual_score=individual_score,\n        )\n        terminal = self.reward_model.terminal_reward(\n            terminal_value, rpn=rpn, individual_score=individual_score\n        )\n        transition = terminal.transition\n        self._emit(\n            "mcts_terminal_result",\n            rpn=rpn,\n            terminal_reward=float(terminal.reward),\n            committed=bool(getattr(transition, "committed", False)),\n            previous_score=getattr(transition, "previous_score", None),\n            resulting_score=getattr(transition, "resulting_score", None),\n            additive_delta=getattr(transition, "additive_delta", None),\n            pool_size=getattr(transition, "pool_size", None),\n            evicted=(getattr(getattr(transition, "evicted", None), "rpn", None)),\n        )\n        terminal_reward = float(terminal.reward)\n''',
    'terminal logging',
)

# Episode completion logging.
s = replace_once(
    s,
    '''        trajectory = PolicyTrajectory(\n            states=tuple(state.token_ids for state in states),\n            actions=tuple(actions),\n            legal_actions=tuple(legal_history),\n            reward=float(total_reward),\n            step_rewards=tuple(step_rewards),\n            terminal_formula_key=terminal_value.canonical_key,\n            terminal_formula_rpn=rpn,\n            pool_changed=bool(terminal.transition.committed),\n        )\n        return (\n''',
    '''        trajectory = PolicyTrajectory(\n            states=tuple(state.token_ids for state in states),\n            actions=tuple(actions),\n            legal_actions=tuple(legal_history),\n            reward=float(total_reward),\n            step_rewards=tuple(step_rewards),\n            terminal_formula_key=terminal_value.canonical_key,\n            terminal_formula_rpn=rpn,\n            pool_changed=bool(terminal.transition.committed),\n        )\n        self._emit(\n            "mcts_episode_done",\n            rpn=rpn,\n            action_count=len(actions),\n            step_rewards=[float(value) for value in step_rewards],\n            returns=[float(value) for value in returns],\n            total_reward=float(total_reward),\n            pool_changed=trajectory.pool_changed,\n        )\n        return (\n''',
    'episode done logging',
)

# Edge-by-edge backprop logging.
s = replace_once(
    s,
    '''            node = self.nodes[node_key]\n            edge = node.edges[token_id]\n            node.virtual_visits = max(0, node.virtual_visits - 1)\n            edge.virtual_visits = max(0, edge.virtual_visits - 1)\n            node.visits += 1\n            edge.visits += 1\n            edge.reward = float(immediate)\n            edge.value_sum += (\n                float(value) if math.isfinite(value) else self.config.invalid_reward\n            )\n''',
    '''            node = self.nodes[node_key]\n            edge = node.edges[token_id]\n            before_q = float(edge.q)\n            before_visits = int(edge.visits)\n            node.virtual_visits = max(0, node.virtual_visits - 1)\n            edge.virtual_visits = max(0, edge.virtual_visits - 1)\n            node.visits += 1\n            edge.visits += 1\n            edge.reward = float(immediate)\n            edge.value_sum += (\n                float(value) if math.isfinite(value) else self.config.invalid_reward\n            )\n            self._emit(\n                "mcts_backprop_edge",\n                token=self.environment.vocabulary.by_id[token_id].name,\n                immediate_reward=float(immediate),\n                cumulative_return=float(value),\n                visits_before=before_visits,\n                visits_after=edge.visits,\n                q_before=before_q,\n                q_after=float(edge.q),\n                node_visits=node.visits,\n            )\n''',
    'dense backprop logging',
)
p.write_text(s)


# ---------------------------------------------------------------------------
# Replay + policy-training tracing
# ---------------------------------------------------------------------------
p = Path('src/flows/riskminer/trainer.py')
s = p.read_text()
s = replace_once(
    s,
    '''        replay = ReplayBuffer(active.replay_capacity)\n        search = RewardDenseRiskMCTS(\n            environment, reward_model, config=active, policy=self.policy\n        ).search()\n        replay.extend(search.trajectories)\n        rewards = [trajectory.reward for trajectory in replay.trajectories]\n''',
    '''        replay = ReplayBuffer(active.replay_capacity)\n        self._emit(\n            "replay_reset",\n            iteration=index,\n            capacity=active.replay_capacity,\n        )\n        search = RewardDenseRiskMCTS(\n            environment,\n            reward_model,\n            config=active,\n            policy=self.policy,\n            on_event=self.on_event,\n        ).search()\n        replay.extend(search.trajectories)\n        replay_records = [\n            {\n                "index": replay_index,\n                "reward": float(trajectory.reward),\n                "step_rewards": [float(value) for value in trajectory.step_rewards],\n                "terminal_rpn": trajectory.terminal_formula_rpn,\n                "pool_changed": trajectory.pool_changed,\n                "actions": [\n                    environment.vocabulary.by_id[action].name\n                    for action in trajectory.actions\n                ],\n            }\n            for replay_index, trajectory in enumerate(replay.trajectories)\n        ]\n        self._emit(\n            "replay_snapshot",\n            iteration=index,\n            capacity=active.replay_capacity,\n            size=len(replay_records),\n            trajectories=replay_records,\n        )\n        rewards = [trajectory.reward for trajectory in replay.trajectories]\n''',
    'trainer replay logging',
)
s = replace_once(
    s,
    '''        for reward in rewards:\n            # Equations 11 and 13 use q_i for trajectory i.  Record the\n            # pre-update threshold, then advance the stochastic quantile\n            # recursion to q_(i+1).\n            trajectory_quantiles.append(float(self.quantile.value))\n            self.quantile = self.quantile.update(float(reward))\n''',
    '''        for trajectory_index, reward in enumerate(rewards):\n            # Equations 11 and 13 use q_i for trajectory i.  Record the\n            # pre-update threshold, then advance the stochastic quantile\n            # recursion to q_(i+1).\n            threshold = float(self.quantile.value)\n            trajectory_quantiles.append(threshold)\n            self.quantile = self.quantile.update(float(reward))\n            self._emit(\n                "replay_quantile_update",\n                iteration=index,\n                trajectory_index=trajectory_index,\n                reward=float(reward),\n                threshold_before=threshold,\n                threshold_after=float(self.quantile.value),\n                selected_for_risk_update=bool(float(reward) <= threshold),\n            )\n''',
    'trainer quantile logging',
)
s = replace_once(
    s,
    '''        losses: list[float] = []\n        for batch in replay.batches(\n            active.policy_batch_size,\n            epochs=active.policy_train_epochs,\n            seed=active.seed + index,\n            shuffle=True,\n            reward_quantiles=trajectory_quantiles,\n        ):\n            self.policy, loss = self.policy.train_step(batch, self.quantile.value)\n            losses.append(float(loss))\n''',
    '''        losses: list[float] = []\n        for batch_index, batch in enumerate(\n            replay.batches(\n                active.policy_batch_size,\n                epochs=active.policy_train_epochs,\n                seed=active.seed + index,\n                shuffle=True,\n                reward_quantiles=trajectory_quantiles,\n            ),\n            1,\n        ):\n            batch_records = []\n            for row, trajectory in enumerate(batch.trajectories):\n                threshold = (\n                    float(batch.reward_quantiles[row])\n                    if batch.reward_quantiles\n                    else float(self.quantile.value)\n                )\n                batch_records.append({\n                    "reward": float(trajectory.reward),\n                    "threshold": threshold,\n                    "selected_for_risk_update": bool(float(trajectory.reward) <= threshold),\n                    "terminal_rpn": trajectory.terminal_formula_rpn,\n                    "pool_changed": trajectory.pool_changed,\n                    "action_count": len(trajectory.actions),\n                })\n            self._emit(\n                "policy_train_batch_start",\n                iteration=index,\n                batch=batch_index,\n                batch_size=len(batch.trajectories),\n                trajectories=batch_records,\n            )\n            self.policy, loss = self.policy.train_step(batch, self.quantile.value)\n            losses.append(float(loss))\n            self._emit(\n                "policy_train_batch_done",\n                iteration=index,\n                batch=batch_index,\n                loss=float(loss),\n            )\n''',
    'trainer batch logging',
)
p.write_text(s)


# ---------------------------------------------------------------------------
# Runner log levels. Default TRACE because this script is diagnostic by design.
# ---------------------------------------------------------------------------
p = Path('scripts/run_riskminer_inputdata.py')
s = p.read_text()
s = replace_once(
    s,
    'RESUME_POLICY = os.environ.get("RISKMINER_RESUME_POLICY", "").strip()\n',
    '''RESUME_POLICY = os.environ.get("RISKMINER_RESUME_POLICY", "").strip()\nLOG_LEVEL = os.environ.get("RISKMINER_LOG_LEVEL", "trace").strip().lower()\nif LOG_LEVEL not in {"summary", "detail", "trace"}:\n    raise ValueError("RISKMINER_LOG_LEVEL must be summary, detail, or trace")\nLOG_LEVEL_RANK = {"summary": 0, "detail": 1, "trace": 2}\nTRACE_EVENTS = {\n    "mcts_node_choice",\n    "mcts_selection_edge",\n    "mcts_rollout_choice",\n    "mcts_rollout_step",\n    "mcts_backprop_edge",\n}\nDETAIL_EVENTS = {\n    "mcts_search_start", "mcts_search_done",\n    "mcts_simulation_start", "mcts_simulation_done",\n    "mcts_selection_done", "mcts_rollout_start", "mcts_rollout_done",\n    "mcts_episode_invalid", "mcts_episode_done",\n    "mcts_candidates_evaluate", "mcts_candidates_scored",\n    "mcts_archive_update", "mcts_terminal_evaluate", "mcts_terminal_result",\n    "replay_reset", "replay_snapshot", "replay_quantile_update",\n    "policy_train_batch_start", "policy_train_batch_done",\n}\n''',
    'runner log-level config',
)
s = replace_once(
    s,
    '''    def event(event_name: str, payload) -> None:\n        progress.emit(event_name, **dict(payload))\n''',
    '''    def event(event_name: str, payload) -> None:\n        required = (\n            2 if event_name in TRACE_EVENTS\n            else 1 if event_name in DETAIL_EVENTS\n            else 0\n        )\n        if LOG_LEVEL_RANK[LOG_LEVEL] >= required:\n            progress.emit(event_name, **dict(payload))\n''',
    'runner event filter',
)
s = replace_once(
    s,
    '''        ridge_recompute_every=RIDGE_RECOMPUTE_EVERY,\n    )\n''',
    '''        ridge_recompute_every=RIDGE_RECOMPUTE_EVERY,\n        log_level=LOG_LEVEL,\n    )\n''',
    'runner start log level',
)
s = replace_once(
    s,
    '''            "policy_learning_rate": POLICY_LEARNING_RATE,\n        },\n''',
    '''            "policy_learning_rate": POLICY_LEARNING_RATE,\n            "log_level": LOG_LEVEL,\n        },\n''',
    'runner report log level',
)
p.write_text(s)


# ---------------------------------------------------------------------------
# Tests: verify MCTS and trainer emit the requested granular records.
# ---------------------------------------------------------------------------
p = Path('tests/flows/riskminer/test_paper_pipeline.py')
s = p.read_text()
if 'test_granular_mcts_and_replay_events_are_emitted' not in s:
    s += '''\n\ndef test_granular_mcts_and_replay_events_are_emitted(tmp_path):\n    environment, config = _tiny_environment(seed=17)\n    events = []\n\n    def on_event(name, payload):\n        events.append((name, payload))\n\n    trainer = RiskSeekingTrainer(\n        vocabulary_size=len(environment.vocabulary),\n        config=config,\n        output_dir=tmp_path,\n        on_event=on_event,\n    )\n    trainer.run_iteration(environment, FakeRewardModel(), iteration=1)\n    names = [name for name, _ in events]\n    required = {\n        "mcts_search_start",\n        "mcts_node_choice",\n        "mcts_selection_edge",\n        "mcts_rollout_done",\n        "mcts_candidates_evaluate",\n        "mcts_candidates_scored",\n        "mcts_terminal_evaluate",\n        "mcts_terminal_result",\n        "mcts_episode_done",\n        "mcts_backprop_edge",\n        "mcts_search_done",\n        "replay_reset",\n        "replay_snapshot",\n        "replay_quantile_update",\n        "policy_train_batch_start",\n        "policy_train_batch_done",\n    }\n    assert required <= set(names)\n\n    node_choice = next(payload for name, payload in events if name == "mcts_node_choice")\n    assert node_choice["edges"]\n    assert {"token", "prior", "q", "visits", "puct"} <= set(node_choice["edges"][0])\n\n    candidates = next(payload for name, payload in events if name == "mcts_candidates_evaluate")\n    assert candidates["candidate_count"] >= 1\n    assert candidates["candidates"][0]["expr"]\n\n    snapshot = next(payload for name, payload in events if name == "replay_snapshot")\n    assert snapshot["size"] == 1\n    assert snapshot["trajectories"][0]["actions"]\n\n    backprop = next(payload for name, payload in events if name == "mcts_backprop_edge")\n    assert backprop["visits_after"] == backprop["visits_before"] + 1\n'''
p.write_text(s)

print('granular logging patch applied')
