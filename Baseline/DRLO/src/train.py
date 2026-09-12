import csv
import json
import os
import random
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf

import Actor
import Critic
import ReplayBuffer
import utils
from Environment import Env


np.set_printoptions(threshold=20)


def _metric_value(metrics, key, default=0.0):
    if metrics is None:
        return default
    return float(metrics.to_dict().get(key, default))


def _profile_suffix(profile_name):
    if not profile_name:
        return ""
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(profile_name)).strip("_")
    return f"_{safe}" if safe else ""


def _iter_weight_profiles(config):
    profiles = config.get("weight_profiles")
    if not profiles:
        return []

    if isinstance(profiles, dict):
        parsed = [(name, weights) for name, weights in profiles.items()]
    else:
        parsed = []
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            name = profile.get("name", f"Profile{len(parsed) + 1}")
            weights = profile.get("weights")
            if weights is not None:
                parsed.append((name, weights))

    run_profiles = config.get("run_weight_profiles")
    if run_profiles:
        if isinstance(run_profiles, str):
            selected = {run_profiles}
        else:
            selected = {str(name) for name in run_profiles}
        parsed = [(name, weights) for name, weights in parsed if name in selected]
        if not parsed:
            raise ValueError(f"No matching weight profiles found for run_weight_profiles={sorted(selected)}")

    return parsed


def save_detailed_allocation(env, deployment, acu_filename, allocation_filename, legacy_filename=None):
    assign = env.Assign
    metrics, sigma, site_loads, deployment = assign.evaluate_system(deployment, return_details=True)
    active = np.flatnonzero(deployment > 0)

    with open(acu_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "BS_ID",
                "Latitude",
                "Longitude",
                "ACU_Count",
                "Capacity",
                "ES_Workload",
                "ES_Utilization_pct",
            ]
        )
        for idx in active:
            capacity = assign.service_capacity * deployment[idx]
            util = site_loads[idx] / capacity if capacity > 0 else 0.0
            writer.writerow(
                [
                    assign.bs_ids[idx],
                    f"{assign.latitudes[idx]:.6f}",
                    f"{assign.longitudes[idx]:.6f}",
                    int(deployment[idx]),
                    f"{capacity:.6f}",
                    f"{site_loads[idx]:.6f}",
                    f"{util * 100.0:.2f}",
                ]
            )

    with open(allocation_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "BS_ID",
                "Latitude",
                "Longitude",
                "Workload",
                "Local_ACU_Count",
                "Serving_ES_ID",
                "Serving_ES_ACU_Count",
                "Dist_to_ES_m",
            ]
        )
        for bs_idx in range(assign.bs_num):
            es_idx = int(sigma[bs_idx])
            if es_idx == -1:
                serving_es = "CLOUD"
                serving_acus = 0
                distance = -1.0
            else:
                serving_es = assign.bs_ids[es_idx]
                serving_acus = int(deployment[es_idx])
                distance = float(assign.dist_matrix[es_idx, bs_idx])
            writer.writerow(
                [
                    assign.bs_ids[bs_idx],
                    f"{assign.latitudes[bs_idx]:.6f}",
                    f"{assign.longitudes[bs_idx]:.6f}",
                    f"{assign.lambda_k[bs_idx]:.8f}",
                    int(deployment[bs_idx]),
                    serving_es,
                    serving_acus,
                    f"{distance:.1f}",
                ]
            )

    if legacy_filename:
        with open(acu_filename, "r", encoding="utf-8") as src, open(
            legacy_filename, "w", newline="", encoding="utf-8"
        ) as dst:
            dst.write(src.read())

    return metrics


def ddpg_play(ddpg_config, profile_name=None, profile_weights=None):
    tf.keras.backend.set_floatx("float64")

    buffer_size = int(ddpg_config.get("buffer_size", 40000))
    batch_size = int(ddpg_config.get("batch_size", 256))
    gamma = float(ddpg_config.get("gamma", 0.9))
    tau = float(ddpg_config.get("tau", 0.001))
    lr_a = float(ddpg_config.get("lr_a", 5e-7))
    lr_c = float(ddpg_config.get("lr_c", 5e-6))

    episode = int(ddpg_config.get("episode", 2000))
    max_step = int(ddpg_config.get("max_step", 30))
    log_start_episode = int(ddpg_config.get("log_start_episode", 10))
    state_dim = int(ddpg_config.get("state_dim", ddpg_config.get("days", 31)))
    explore = float(ddpg_config.get("explore", 0.9))
    epsilon_rate = float(ddpg_config.get("epsilon_rate", 0.0005))
    explore_steps = explore * episode

    data_path = ddpg_config.get("data_path", "bs_statistics_all_12files.csv")
    requested_bs_num = int(ddpg_config.get("bs_num", 0))
    bs_num = utils.resolve_bs_num(data_path, requested_bs_num)
    if requested_bs_num > 0 and requested_bs_num != bs_num:
        print(f"Requested bs_num={requested_bs_num}, using available rows={bs_num}.")

    es_ratio = float(ddpg_config.get("es_ratio", 0.0))
    acu_budget = int(ddpg_config.get("acu_budget", round(bs_num * es_ratio)))
    n_max = int(ddpg_config.get("n_max", 3))
    d_max_m = float(ddpg_config.get("d_max_m", 1500.0))
    service_capacity = ddpg_config.get("service_capacity", 11000.0)
    weights = profile_weights if profile_weights is not None else ddpg_config.get("weights", None)
    profile_label = profile_name if profile_name is not None else ddpg_config.get("profile_name", "")
    suffix = _profile_suffix(profile_label)
    capacity_constrained_mapping = bool(ddpg_config.get("capacity_constrained_mapping", False))
    site_idle_energy = float(ddpg_config.get("site_idle_energy", 0.0))
    acu_static_energy = float(ddpg_config.get("acu_static_energy", 1.0))
    acu_dynamic_energy = float(ddpg_config.get("acu_dynamic_energy", 1.0))
    dynamic_energy_exponent = float(ddpg_config.get("dynamic_energy_exponent", 1.0))
    energy_efficiency_bounds = ddpg_config.get("energy_efficiency_bounds", "vneqts")
    load_balance_positive_only = bool(ddpg_config.get("load_balance_positive_only", True))

    acu_budget = min(max(1, acu_budget), bs_num * n_max)
    init_vector = utils.random_deployment(bs_num, acu_budget, n_max)

    print(
        "DRLO ACU setup: "
        f"profile={profile_label or 'single'}, N={bs_num}, M={acu_budget}, "
        f"n_max={n_max}, Dmax={d_max_m:.1f}m, C={service_capacity}, "
        f"mapping={'capacity-constrained' if capacity_constrained_mapping else 'nearest-within-radius'}, "
        f"energy=(site_idle={site_idle_energy}, acu_static={acu_static_energy}, "
        f"acu_dynamic={acu_dynamic_energy}, exponent={dynamic_energy_exponent}, "
        f"bounds={energy_efficiency_bounds}), weights={weights}, data={data_path}"
    )

    env = Env(
        bs_num,
        acu_budget,
        data_path,
        state_dim,
        init_vector,
        n_max=n_max,
        d_max_m=d_max_m,
        weights=weights,
        service_capacity=service_capacity,
        capacity_constrained_mapping=capacity_constrained_mapping,
        site_idle_energy=site_idle_energy,
        acu_static_energy=acu_static_energy,
        acu_dynamic_energy=acu_dynamic_energy,
        dynamic_energy_exponent=dynamic_energy_exponent,
        energy_efficiency_bounds=energy_efficiency_bounds,
        load_balance_positive_only=load_balance_positive_only,
    )
    bs_num = env.bs_num
    action_dim = bs_num

    dirs_to_create = ["src", "train_log", "es_place", "img/loss", "img/reward", "img/delay", "img/workload_bias"]
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)

    buff = ReplayBuffer.ReplayBuffer(buffer_size)
    actor = Actor.Actor(state_dim, action_dim, lr_a)
    actor_target = Actor.Actor(state_dim, action_dim, lr_a)
    critic = Critic.Critic(state_dim, action_dim, lr_c)
    critic_target = Critic.Critic(state_dim, action_dim, lr_c)

    actor_target.model.set_weights(actor.model.get_weights())
    critic_target.model.set_weights(critic.model.get_weights())

    model_tag = f"{episode}iters_{max_step}steps_{bs_num}bs_{acu_budget}acu{suffix}"
    actor_model_path = f"src/actormodel_{model_tag}.weights.h5"
    critic_model_path = f"src/criticmodel_{model_tag}.weights.h5"

    global_best_reward = -float("inf")
    global_best_action = None
    global_best_metrics = None
    start_time_alg = time.time()

    convergence_filename = f"convergence_DRLO{suffix}.csv"
    with open(convergence_filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Episode", "Profile", "F", "PhiSite", "PhiTraf", "B", "Ee", "E", "AvgUtil", "CF"])

        print("Experiment Start.")
        for ep in range(episode):
            state, _, _, _ = env.reset(0)
            total_reward = 0.0
            total_loss = 0.0
            step = 0

            for t in range(max_step):
                state_batch = state.reshape(1, state_dim)
                action_origin = actor.model.predict(state_batch, verbose=0)
                action = action_origin[0]

                if ep <= explore_steps:
                    epsilon = abs(abs(action).sum()) * epsilon_rate
                    noise = np.random.normal(0.0, 0.5, action_dim) * epsilon
                    tanh_y = float(ddpg_config.get("tanh_y", 20.0))
                    action = np.where(
                        (action + noise > -tanh_y) & (action + noise < tanh_y),
                        action + noise,
                        action - noise,
                    )

                next_state, phi_site, phi_traf, reward, done, balance, energy, cloud_fallback = env.step(action, 0)
                del phi_site, phi_traf, balance, energy, cloud_fallback
                next_state = next_state.reshape(state_dim)

                if t == max_step - 1:
                    done = True

                buff.add(state.squeeze(), action, reward, next_state, done)
                states, actions, rewards, next_states, dones = buff.getBatch(batch_size)

                target_actions = actor_target.model.predict(next_states, verbose=0)
                target_q_values = critic_target.model.predict([next_states, target_actions], verbose=0)
                yi = compute_yi(rewards, target_q_values, dones, gamma)
                loss = critic.train(states, actions, yi, lr_c)

                a_for_grads = actor.model.predict(states, verbose=0)
                a_grads = critic.q_grads(states, a_for_grads)
                actor.train(states, a_grads, lr_a)
                target_update(actor, actor_target, critic, critic_target, tau)

                metrics = env.Assign.last_metrics
                if reward > global_best_reward:
                    global_best_reward = reward
                    global_best_action = action.copy()
                    global_best_metrics = metrics

                total_reward += reward
                total_loss += float(loss.numpy() if hasattr(loss, "numpy") else loss)
                step += 1
                state = next_state

            if ep % 5 == 0:
                actor.model.save_weights(actor_model_path, overwrite=True)
                critic.model.save_weights(critic_model_path, overwrite=True)

            if ep >= log_start_episode:
                writer.writerow(
                    [
                        ep,
                        profile_label or "single",
                        global_best_reward,
                        _metric_value(global_best_metrics, "phi_site"),
                        _metric_value(global_best_metrics, "phi_traf"),
                        _metric_value(global_best_metrics, "balance"),
                        _metric_value(global_best_metrics, "energy_eff"),
                        _metric_value(global_best_metrics, "total_energy"),
                        _metric_value(global_best_metrics, "avg_util"),
                        int(_metric_value(global_best_metrics, "cloud_fallback", bs_num)),
                    ]
                )
                csv_file.flush()

            if ep >= log_start_episode and ep % 10 == 0:
                avg_reward = total_reward / max(step, 1)
                avg_loss = total_loss / max(step, 1)
                best_avg_util = _metric_value(global_best_metrics, "avg_util")
                print(
                    f"Iteration {ep}: best F={global_best_reward:.6f}, "
                    f"avg_reward={avg_reward:.6f}, avg_util={best_avg_util * 100.0:.4f}%, "
                    f"loss={avg_loss:.6f}"
                )

            if total_reward / max(step, 1) > 0 and ep > explore_steps:
                lr_a = lr_a * 0.9 ** (ep / 100)

    end_time_alg = time.time()
    time_cost_mins = (end_time_alg - start_time_alg) / 60.0

    if global_best_action is None:
        global_best_action = action

    final_deployment = utils.transform_action(global_best_action, acu_budget, n_max)
    final_metrics = save_detailed_allocation(
        env,
        final_deployment,
        f"acu_deployment_DRLO{suffix}.csv",
        f"bs_es_allocation_DRLO{suffix}.csv",
        legacy_filename=f"DRLO_allocation_rep1{suffix}.csv",
    )

    print(f"\n========== DRLO ACU Final Result ({profile_label or 'single'}) ==========")
    print(f"Composite fitness F: {final_metrics.fitness:.6f}")
    print(f"Site coverage Phi_site: {final_metrics.phi_site * 100.0:.4f}%")
    print(f"Traffic coverage Phi_traf: {final_metrics.phi_traf * 100.0:.4f}%")
    print(f"Load balance B: {final_metrics.balance:.6f}")
    print(f"Energy efficiency Ee: {final_metrics.energy_eff:.6f}")
    print(f"Total energy E: {final_metrics.total_energy:.6f}")
    print(f"Average utilization: {final_metrics.avg_util * 100.0:.4f}%")
    print(f"Cloud fallback BS: {final_metrics.cloud_fallback} / {env.bs_num}")
    print(f"Active sites: {final_metrics.active_sites} / {env.bs_num}")
    print(f"DRLO runtime: {time_cost_mins:.2f} min")
    print(
        "Saved files: "
        f"{convergence_filename}, acu_deployment_DRLO{suffix}.csv, "
        f"bs_es_allocation_DRLO{suffix}.csv"
    )
    return final_metrics


def run_configured_experiments(ddpg_config):
    profiles = _iter_weight_profiles(ddpg_config)
    if not profiles:
        return [("single", ddpg_play(ddpg_config))]

    results = []
    for profile_name, weights in profiles:
        profile_config = dict(ddpg_config)
        profile_config["weights"] = weights
        profile_config["profile_name"] = profile_name
        print(f"\n########## Running {profile_name}: weights={weights} ##########")
        metrics = ddpg_play(profile_config, profile_name=profile_name, profile_weights=weights)
        results.append((profile_name, weights, metrics))

    summary_suffix = ""
    if ddpg_config.get("run_weight_profiles"):
        summary_suffix = _profile_suffix("_".join(name for name, _, _ in results))
    summary_filename = f"weight_profile_summary_DRLO{summary_suffix}.csv"
    with open(summary_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Profile",
                "w_site",
                "w_traffic",
                "w_balance",
                "w_energy",
                "F",
                "PhiSite",
                "PhiTraf",
                "B",
                "Ee",
                "E",
                "AvgUtil",
                "CF",
                "ActiveSites",
            ]
        )
        for profile_name, weights, metrics in results:
            w = np.asarray(weights, dtype=np.float64).ravel()
            if w.size != 4:
                w = np.zeros(4, dtype=np.float64)
            writer.writerow(
                [
                    profile_name,
                    w[0],
                    w[1],
                    w[2],
                    w[3],
                    metrics.fitness,
                    metrics.phi_site,
                    metrics.phi_traf,
                    metrics.balance,
                    metrics.energy_eff,
                    metrics.total_energy,
                    metrics.avg_util,
                    metrics.cloud_fallback,
                    metrics.active_sites,
                ]
            )
    print(f"Saved profile summary: {summary_filename}")
    return results


def compute_yi(rewards, target_q_values, dones, gamma):
    yi = np.asarray(target_q_values).copy()
    for i in range(target_q_values.shape[0]):
        if dones[i]:
            yi[i] = rewards[i]
        else:
            yi[i] = gamma * target_q_values[i] + rewards[i]
    return yi


def target_update(actor, actor_target, critic, critic_target, tau):
    actor_weights = actor.model.get_weights()
    target_actor_weights = actor_target.model.get_weights()
    critic_weights = critic.model.get_weights()
    target_critic_weights = critic_target.model.get_weights()

    for i in range(len(actor_weights)):
        target_actor_weights[i] = tau * actor_weights[i] + (1.0 - tau) * target_actor_weights[i]

    for i in range(len(critic_weights)):
        target_critic_weights[i] = tau * critic_weights[i] + (1.0 - tau) * target_critic_weights[i]

    actor_target.model.set_weights(target_actor_weights)
    critic_target.model.set_weights(target_critic_weights)


if __name__ == "__main__":
    with open("parameter.json", encoding="utf-8") as jconfig:
        esplace_config = json.load(jconfig)
    run_configured_experiments(esplace_config)
