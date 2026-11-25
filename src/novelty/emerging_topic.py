"""Stage 3.10 Part 3: Separate Emerging-Topic Detection Subsystem.

Distinct from Event Clustering (assigning articles to events) and Sentence Novelty
(detecting new factual propositions within an event), this module detects when an
entirely new coherent topic emerges in a chronological article stream, and explicitly
separates:
  1. Per-article cosine novelty (isolated outliers without topic clustering)
  2. EventPulse semantic incremental clustering (tracking new-cluster formation vs absorption)
  3. Streaming density micro-cluster detector (DBSTREAM-style with fading & absorption tracking)
  4. Rolling-window coherent emergence detector (windowed cluster burst + enrichment)
"""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class EmergenceEvent:
    detector: str
    step: int
    article_id: Any
    cluster_id: Optional[int]
    mode: str  # 'isolated_flag', 'new_cluster_formation', 'absorption_into_existing', 'window_emergence'
    is_synthetic: bool
    scenario_id: Optional[str]


class StreamingCosineNoveltyDetector:
    """Sliding-window cosine novelty baseline (tracks max cosine to recent window)."""

    def __init__(self, window_size: int = 1000, sim_threshold: float = 0.40):
        self.window_size = window_size
        self.sim_threshold = sim_threshold
        self._buf: Optional[np.ndarray] = None
        self._count = 0
        self._pos = 0
        self.flags: List[Dict[str, Any]] = []

    def calibrate_percentile(self, warmup_vecs: np.ndarray, percentile: float = 5.0) -> float:
        """Estimate p5 nearest-neighbor max cosine similarity cleanly on warmup control stream."""
        n = warmup_vecs.shape[0]
        if n < 50:
            return self.sim_threshold
        buf = np.empty((self.window_size, warmup_vecs.shape[1]), dtype=np.float32)
        cnt, pos = 0, 0
        max_sims = []
        for i in range(n):
            v = warmup_vecs[i]
            if cnt >= 50:
                max_sims.append(float(np.max(buf[:cnt] @ v)))
            buf[pos] = v
            pos = (pos + 1) % self.window_size
            cnt = min(cnt + 1, self.window_size)
        self.sim_threshold = float(np.percentile(max_sims, percentile))
        return self.sim_threshold

    def process_one(self, step: int, article_id: Any, vec: np.ndarray, is_synthetic: bool, scenario_id: Optional[str]) -> Dict[str, Any]:
        if self._buf is None:
            self._buf = np.empty((self.window_size, vec.shape[0]), dtype=np.float32)

        flagged = False
        max_sim = 0.0
        if self._count > 0:
            max_sim = float(np.max(self._buf[:self._count] @ vec))
            if max_sim < self.sim_threshold:
                flagged = True
                self.flags.append({
                    "step": step,
                    "article_id": article_id,
                    "max_sim": max_sim,
                    "is_synthetic": is_synthetic,
                    "scenario_id": scenario_id,
                })

        self._buf[self._pos] = vec
        self._pos = (self._pos + 1) % self.window_size
        self._count = min(self._count + 1, self.window_size)
        return {
            "flagged": flagged,
            "max_sim": max_sim,
            "cluster_id": None,
            "seeded_before_onset": False,
            "mode": "isolated_flag" if flagged else "normal",
        }


class EventPulseClusterEmergenceTracker:
    """Applies EventPulse semantic clustering to stream emergence and separates new clusters vs absorption."""

    def __init__(self, match_threshold: float = 0.55, max_active: int = 2500):
        self.match_threshold = match_threshold
        self.max_active = max_active
        self.sums: List[np.ndarray] = []
        self._cent_buf: Optional[np.ndarray] = None  # (max_active, D) preallocated float32
        self._count = 0
        self.cluster_ids: List[int] = []
        self.seed_is_synthetic: List[bool] = []
        self.seed_step: List[int] = []
        self.last_step: List[int] = []
        self.members: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self._next_cid = 0

    def process_one(
        self,
        step: int,
        article_id: Any,
        vec: np.ndarray,
        is_synthetic: bool,
        scenario_id: Optional[str],
        onset_step: int,
    ) -> Dict[str, Any]:
        if self._cent_buf is None:
            self._cent_buf = np.empty((self.max_active, vec.shape[0]), dtype=np.float32)

        if self._count == 0:
            cid = self._next_cid
            self._next_cid += 1
            self.sums = [vec.copy()]
            self._cent_buf[0] = vec
            self._count = 1
            self.cluster_ids = [cid]
            self.seed_is_synthetic = [is_synthetic]
            self.seed_step = [step]
            self.last_step = [step]
            self.members[cid].append({"step": step, "is_synthetic": is_synthetic, "scenario_id": scenario_id})
            return {"cluster_id": cid, "is_new_cluster": True, "seeded_before_onset": (step < onset_step), "mode": "new_cluster_formation"}

        sims = self._cent_buf[: self._count] @ vec
        best_pos = int(np.argmax(sims))
        best_sim = float(sims[best_pos])

        if best_sim >= self.match_threshold:
            cid = self.cluster_ids[best_pos]
            self.sums[best_pos] += vec
            nrm = float(np.linalg.norm(self.sums[best_pos]))
            self._cent_buf[best_pos] = self.sums[best_pos] / nrm if nrm > 1e-12 else vec
            self.last_step[best_pos] = step
            self.members[cid].append({"step": step, "is_synthetic": is_synthetic, "scenario_id": scenario_id})
            seeded_before = self.seed_step[best_pos] < onset_step
            return {
                "cluster_id": cid,
                "is_new_cluster": False,
                "seeded_before_onset": seeded_before,
                "mode": "absorption_into_existing" if seeded_before else "new_cluster_growth",
            }
        else:
            cid = self._next_cid
            self._next_cid += 1
            if self._count >= self.max_active:
                pos = int(np.argmin(self.last_step))
                self.sums[pos] = vec.copy()
                self._cent_buf[pos] = vec
                self.cluster_ids[pos] = cid
                self.seed_is_synthetic[pos] = is_synthetic
                self.seed_step[pos] = step
                self.last_step[pos] = step
            else:
                pos = self._count
                self.sums.append(vec.copy())
                self._cent_buf[pos] = vec
                self._count += 1
                self.cluster_ids.append(cid)
                self.seed_is_synthetic.append(is_synthetic)
                self.seed_step.append(step)
                self.last_step.append(step)
            self.members[cid].append({"step": step, "is_synthetic": is_synthetic, "scenario_id": scenario_id})
            return {
                "cluster_id": cid,
                "is_new_cluster": True,
                "seeded_before_onset": (step < onset_step),
                "mode": "new_cluster_formation",
            }


class StreamingMicroClusterDetector:
    """DBSTREAM-style vectorized streaming density micro-cluster detector (Euclidean radius on L2 vectors).

    Note: for unit-normalized embeddings, ||u - v||^2 = 2 - 2 cos(u, v), so
    euclidean threshold r = 1.0 corresponds strictly to cosine = 0.50.
    """

    def __init__(self, radius_threshold: float = 1.0, fading_factor: float = 0.005, min_weight: float = 0.15, cleanup_interval: int = 200, max_active: int = 4000):
        self.radius_threshold = radius_threshold
        self.cos_equiv = 1.0 - 0.5 * (radius_threshold ** 2)
        self.fading_factor = fading_factor
        self.min_weight = min_weight
        self.cleanup_interval = cleanup_interval
        self.max_active = max_active
        self._buf: Optional[np.ndarray] = None
        self._count = 0
        self.weights: List[float] = []
        self.mc_ids: List[int] = []
        self.seed_step: List[int] = []
        self.last_step: List[int] = []
        self.members: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self._next_id = 0

    def process_one(
        self,
        step: int,
        article_id: Any,
        vec: np.ndarray,
        is_synthetic: bool,
        scenario_id: Optional[str],
        onset_step: int,
    ) -> Dict[str, Any]:
        if self._buf is None:
            self._buf = np.empty((self.max_active, vec.shape[0]), dtype=np.float32)

        # Periodic fading cleanup
        if step > 0 and (step % self.cleanup_interval == 0) and self._count > 0:
            keep = []
            for idx in range(self._count):
                age = step - self.last_step[idx]
                decayed = self.weights[idx] * math.exp(-self.fading_factor * age)
                self.weights[idx] = decayed
                if decayed >= self.min_weight:
                    keep.append(idx)
            if len(keep) < self._count:
                if keep:
                    self._buf[: len(keep)] = self._buf[keep]
                self._count = len(keep)
                self.weights = [self.weights[i] for i in keep]
                self.mc_ids = [self.mc_ids[i] for i in keep]
                self.seed_step = [self.seed_step[i] for i in keep]
                self.last_step = [self.last_step[i] for i in keep]

        if self._count == 0:
            mc_id = self._next_id
            self._next_id += 1
            self._buf[0] = vec
            self._count = 1
            self.weights = [1.0]
            self.mc_ids = [mc_id]
            self.seed_step = [step]
            self.last_step = [step]
            self.members[mc_id].append({"step": step, "is_synthetic": is_synthetic, "scenario_id": scenario_id})
            return {"cluster_id": mc_id, "is_new_cluster": True, "seeded_before_onset": (step < onset_step), "mode": "new_cluster_formation"}

        sims = self._buf[: self._count] @ vec
        best_pos = int(np.argmax(sims))
        if float(sims[best_pos]) >= self.cos_equiv:
            mc_id = self.mc_ids[best_pos]
            age = step - self.last_step[best_pos]
            w_prev = self.weights[best_pos] * math.exp(-self.fading_factor * age)
            w_new = w_prev + 1.0
            new_center = (w_prev * self._buf[best_pos] + vec) / w_new
            nrm = float(np.linalg.norm(new_center))
            self._buf[best_pos] = new_center / nrm if nrm > 1e-12 else vec
            self.weights[best_pos] = w_new
            self.last_step[best_pos] = step
            self.members[mc_id].append({"step": step, "is_synthetic": is_synthetic, "scenario_id": scenario_id})
            seeded_before = self.seed_step[best_pos] < onset_step
            return {
                "cluster_id": mc_id,
                "is_new_cluster": False,
                "seeded_before_onset": seeded_before,
                "mode": "absorption_into_existing" if seeded_before else "new_cluster_growth",
            }
        else:
            mc_id = self._next_id
            self._next_id += 1
            if self._count >= self.max_active:
                pos = int(np.argmin(self.last_step))
                self._buf[pos] = vec
                self.weights[pos] = 1.0
                self.mc_ids[pos] = mc_id
                self.seed_step[pos] = step
                self.last_step[pos] = step
            else:
                self._buf[self._count] = vec
                self._count += 1
                self.weights.append(1.0)
                self.mc_ids.append(mc_id)
                self.seed_step.append(step)
                self.last_step.append(step)
            self.members[mc_id].append({"step": step, "is_synthetic": is_synthetic, "scenario_id": scenario_id})
            return {"cluster_id": mc_id, "is_new_cluster": True, "seeded_before_onset": (step < onset_step), "mode": "new_cluster_formation"}


class RollingWindowEmergenceDetector:
    """Rolling-window coherent emergence detector (BERTrend-inspired burst-vs-baseline concept).

    Maintains a current sliding window W_cur of size W_cur_len and a previous reference window W_prev.
    Clustered topic signals inside W_cur that (a) reach min_cluster_size >= N, (b) exhibit low similarity
    to W_prev (< history_max_sim_cutoff), and (c) show sharp window growth rate trigger window-level emergence.
    """

    def __init__(
        self,
        cur_window: int = 400,
        prev_window: int = 1000,
        cluster_sim: float = 0.56,
        history_sim_cutoff: float = 0.58,
    ):
        self.cur_window = cur_window
        self.prev_window = prev_window
        self.cluster_sim = cluster_sim
        self.history_sim_cutoff = history_sim_cutoff
        self.cur_vecs: deque = deque(maxlen=cur_window)
        self._prev_buf: Optional[np.ndarray] = None
        self._prev_count = 0
        self._prev_pos = 0
        self.local_clusters: List[Tuple[int, np.ndarray, List[Dict[str, Any]]]] = []
        self._next_lid = 0

    def process_one(
        self,
        step: int,
        article_id: Any,
        vec: np.ndarray,
        is_synthetic: bool,
        scenario_id: Optional[str],
        onset_step: int,
    ) -> Dict[str, Any]:
        if self._prev_buf is None:
            self._prev_buf = np.empty((self.prev_window, vec.shape[0]), dtype=np.float32)

        if len(self.cur_vecs) == self.cur_window:
            self._prev_buf[self._prev_pos] = self.cur_vecs[0]
            self._prev_pos = (self._prev_pos + 1) % self.prev_window
            self._prev_count = min(self._prev_count + 1, self.prev_window)
        self.cur_vecs.append(vec)

        hist_sim = 0.0
        if self._prev_count >= 50:
            hist_sim = float(np.max(self._prev_buf[: self._prev_count] @ vec))

        # Only form emergent window-topics for articles genuinely distinct from previous window
        if hist_sim >= self.history_sim_cutoff:
            return {"cluster_id": None, "is_new_cluster": False, "seeded_before_onset": True, "mode": "absorbed_into_prev_window"}

        active_window_start = max(0, step - self.cur_window)
        # Prune expired clusters periodically so list stays O(cur_window)
        if step > 0 and step % 400 == 0 and len(self.local_clusters) > 500:
            self.local_clusters = [c for c in self.local_clusters if c[2][-1]["step"] >= active_window_start]

        best_idx = -1
        best_s = -1.0
        for idx, (lid, cent, mems) in enumerate(self.local_clusters):
            if mems[-1]["step"] < active_window_start:
                continue
            s = float(cent @ vec)
            if s >= self.cluster_sim and s > best_s:
                best_s = s
                best_idx = idx

        item = {"step": step, "is_synthetic": is_synthetic, "scenario_id": scenario_id}
        if best_idx >= 0:
            lid, cent, mems = self.local_clusters[best_idx]
            mems.append(item)
            n = len(mems)
            new_cent = ((n - 1) * cent + vec) / n
            nrm = float(np.linalg.norm(new_cent))
            self.local_clusters[best_idx] = (lid, new_cent / nrm if nrm > 1e-12 else vec, mems)
            seeded_before = mems[0]["step"] < onset_step
            return {
                "cluster_id": lid,
                "is_new_cluster": False,
                "seeded_before_onset": seeded_before,
                "mode": "absorption_into_existing" if seeded_before else "window_emergence",
            }
        else:
            lid = self._next_lid
            self._next_lid += 1
            self.local_clusters.append((lid, vec.copy(), [item]))
            return {
                "cluster_id": lid,
                "is_new_cluster": True,
                "seeded_before_onset": (step < onset_step),
                "mode": "window_emergence",
            }
