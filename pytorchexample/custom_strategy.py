"""pytorchexample: A Flower / PyTorch app."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from flwr.app import ArrayRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg


class ReputationFedAvg(FedAvg):
    def __init__(
            self,
            decay_factor: float = 0.8,
            min_reputation_threshold: float = 0.2,
            arrayrecord_key: str = "arrays",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.decay_factor = decay_factor
        self.min_reputation_threshold = min_reputation_threshold
        self.reputation_scores: Dict[str, float] = {}
        self.arrayrecord_key = arrayrecord_key
        self.current_global_weights: Optional[List[np.ndarray]] = None

    def start(self, grid, initial_arrays, train_config, num_rounds):
        self.current_global_weights = initial_arrays.to_numpy_ndarrays()
        return super().start(
            grid=grid,
            initial_arrays=initial_arrays,
            train_config=train_config,
            num_rounds=num_rounds,
        )

    def aggregate_train(
            self,
            server_round: int,
            replies: List[RecordDict],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        if not replies:
            return None, None

        valid_replies = [
            reply
            for reply in replies
            if reply.has_content() and self.arrayrecord_key in reply.content
        ]

        if not valid_replies:
            return None, None

        # -------------------------------------------------------------
        # SORT REPLIES BY NODE ID
        # -------------------------------------------------------------
        valid_replies = sorted(
            valid_replies,
            key=lambda r: str(r.metadata.src_node_id)
        )

        client_updates = []
        node_ids = []
        num_examples = []

        for reply in valid_replies:
            node_id = str(reply.metadata.src_node_id)
            node_ids.append(node_id)

            array_record: ArrayRecord = reply.content[self.arrayrecord_key]
            weights = array_record.to_numpy_ndarrays()

            # Delta = Client Weights - Current Global Weights
            if self.current_global_weights is not None:
                delta = [w - g for w, g in zip(weights, self.current_global_weights)]
            else:
                delta = weights

            client_updates.append(delta)

            metrics = reply.content.get("metrics", MetricRecord())
            num_examples.append(metrics.get("num_examples", 1))

            if node_id not in self.reputation_scores:
                self.reputation_scores[node_id] = 1.0

        num_clients = len(node_ids)
        flat_updates = np.array([
            np.concatenate([p.ravel() for p in update]) for update in client_updates
        ])

        # --- DYNAMIC ATTACKER DETECTION (OUTLIER ESTIMATION) ---
        distances = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = np.linalg.norm(flat_updates[i] - flat_updates[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute average distance from each client to all other clients
        mean_distances = np.mean(distances, axis=1)

        # Automatically detect outliers using Median Absolute Deviation (MAD)
        median_dist = np.median(mean_distances)
        mad = np.median(np.abs(mean_distances - median_dist))

        # Dynamic threshold for anomaly detection
        threshold = median_dist + 2.5 * (mad if mad > 0 else np.std(mean_distances))

        # Dynamically estimate number of malicious nodes based on anomaly score
        detected_malicious = [i for i, d in enumerate(mean_distances) if d > threshold]
        estimated_num_malicious = min(max(0, len(detected_malicious)), num_clients - 2)

        # Safeguards for k and m_top calculation
        m_top = max(1, num_clients - estimated_num_malicious)
        k = max(1, m_top - 2)

        krum_scores = []
        for i in range(num_clients):
            sorted_dists = np.sort(distances[i])
            effective_k = min(k, num_clients - 1)
            score = np.sum(sorted_dists[1: effective_k + 1])
            krum_scores.append(score)

        # -------------------------------------------------------------
        # DETERMINISTIC TIE-BREAKING
        # Sort by (score, node_id) so identical scores break ties predictably
        # -------------------------------------------------------------
        scored_clients = sorted(
            enumerate(krum_scores),
            key=lambda item: (item[1], node_ids[item[0]])
        )
        clean_indices = set(idx for idx, _ in scored_clients[:m_top])

        # Print logs in sorted order
        quality_scores = {}
        print(f"\n--- Round {server_round} Dynamic Krum Filtering (Est. Malicious: {estimated_num_malicious}) ---")
        for idx, node_id in enumerate(node_ids):
            is_clean = idx in clean_indices
            quality_scores[node_id] = 1.0 if is_clean else 0.0

            # EMA Reputation update
            old_score = self.reputation_scores[node_id]
            q_score = quality_scores[node_id]
            self.reputation_scores[node_id] = (
                    self.decay_factor * old_score + (1.0 - self.decay_factor) * q_score
            )

            print(
                f"Node {node_id:<10} | Krum Score: {krum_scores[idx]:.2f} | "
                f"Status: {'ACCEPTED' if is_clean else 'REJECTED'} | "
                f"Reputation: {old_score:.3f} -> {self.reputation_scores[node_id]:.3f}"
            )

        # --- WEIGHTED AGGREGATION ---
        weighted_weights = []
        total_reputation_weight = 0.0

        raw_weights_list = [
            reply.content[self.arrayrecord_key].to_numpy_ndarrays()
            for reply in valid_replies
        ]

        for idx, (node_id, weights, n_samples) in enumerate(
                zip(node_ids, raw_weights_list, num_examples)
        ):
            rep = self.reputation_scores[node_id]

            if idx not in clean_indices:
                continue

            if rep < self.min_reputation_threshold: # ignores the node if quality is low enough
                print(f"[Round {server_round}] Ignored Node {node_id} (Low Rep {rep:.2f})")
                continue

            effective_weight = n_samples * rep
            weighted_weights.append((weights, effective_weight))
            total_reputation_weight += effective_weight

        if not weighted_weights or total_reputation_weight == 0:
            print(f"[Round {server_round}] Warning: All client updates were rejected!")
            return None, None

        # Layer-wise Aggregation
        num_layers = len(raw_weights_list[0])
        aggregated_ndarrays = []

        for layer_idx in range(num_layers):
            layer_sum = sum(
                weights[layer_idx] * weight for weights, weight in weighted_weights
            )
            aggregated_ndarrays.append(layer_sum / total_reputation_weight)

        parameter_keys = list(valid_replies[0].content[self.arrayrecord_key].keys())
        state_dict = {
            key: torch.from_numpy(arr)
            for key, arr in zip(parameter_keys, aggregated_ndarrays)
        }
        aggregated_arrays = ArrayRecord.from_torch_state_dict(state_dict)

        metrics_dict = {
            f"rep_{node_id}": score for node_id, score in self.reputation_scores.items()
        }
        aggregated_metrics = MetricRecord(metrics_dict)
        self.current_global_weights = aggregated_ndarrays
        return aggregated_arrays, aggregated_metrics