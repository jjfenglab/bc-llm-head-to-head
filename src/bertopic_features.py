"""Cluster comparative keyphrases via BERTopic to build topic-level features.

Keyphrases produced by `extract_keyphrases.py` have the form
"<property>: A > B" or "<property>: B > A". We partition keyphrases by that
sign suffix, fit a separate BERTopic model to each partition (configured for
online learning so MH iterations can `partial_fit` new keyphrases without a
full refit), and emit a binary (n_obs, n_topics_total) feature matrix with
namespaced topic IDs so "A>B" and "B>A" topics never share a column.

The BERTopic noise cluster (topic -1) is dropped.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import OnlineCountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA

_SIGN_RE = re.compile(r":\s*(A\s*>\s*B|B\s*>\s*A)\s*$", re.IGNORECASE)


class _Float64IncrementalPCA(IncrementalPCA):
    """IncrementalPCA that casts to float64 so BERTopic/HDBSCAN internals are happy."""

    def fit_transform(self, X, y=None):
        return super().fit_transform(X, y).astype(np.float64)

    def transform(self, X):
        return super().transform(X).astype(np.float64)


def _load_embedding_model(name: str) -> SentenceTransformer:
    """Load a SentenceTransformer, preferring the local HF cache.

    First tries `local_files_only=True` so a cached model loads without
    hitting the network. Falls back to a normal download on cache miss.
    """
    try:
        model = SentenceTransformer(name, local_files_only=True)
        logging.info("Loaded embedding model %s from local cache", name)
        return model
    except Exception as e:
        logging.info("Local embedding model %s not found (%s); downloading", name, e)
        return SentenceTransformer(name)


def _parse_sign(phrase: str) -> str | None:
    m = _SIGN_RE.search(phrase)
    if not m:
        return None
    tag = m.group(1).upper().replace(" ", "")
    return "A>B" if tag == "A>B" else "B>A"


def _split_keyphrases(text: str) -> List[str]:
    return [p.strip() for p in str(text).split(",") if p.strip()]


def _per_obs_phrases(llm_outputs: pd.Series) -> List[List[str]]:
    return [_split_keyphrases(x) for x in llm_outputs.fillna("")]


def _make_online_bertopic(
    n_topics: int, seed: int, embedder: SentenceTransformer, num_pca: int = 5
) -> BERTopic:
    """Build a BERTopic configured for online learning via partial_fit.

    Mirrors LLM-prior-monitoring/src/llm_prior_monitoring/bertopic_state.py.
    """
    umap_model = _Float64IncrementalPCA(n_components=num_pca)
    cluster_model = MiniBatchKMeans(n_clusters=n_topics, random_state=seed)
    vectorizer_model = OnlineCountVectorizer(stop_words="english")
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        embedding_model=embedder,
    )


class SignedBertopicModel:
    """Online-learnable BERTopic pair, one model per sign partition.

    Call `partial_fit(data_df, text_col)` to incorporate a new batch of
    keyphrases. State retained across calls: `all_phrases_{ab,ba}` and the
    accumulated `all_topics_{ab,ba}` topic assignments (one int per phrase,
    in insertion order), matching the pattern in bertopic_state.py.
    """

    def __init__(
        self,
        n_topics: int = 20,
        seed: int = 0,
        embedding_model: str = "all-MiniLM-L6-v2",
        num_pca: int = 5,
    ):
        self.n_topics = n_topics
        self.seed = seed
        self.embedding_model_name = embedding_model
        self._embedder = _load_embedding_model(embedding_model)
        self.model_ab: BERTopic = _make_online_bertopic(n_topics, seed, self._embedder, num_pca)
        self.model_ba: BERTopic = _make_online_bertopic(n_topics, seed, self._embedder, num_pca)
        self.all_phrases_ab: List[str] = []
        self.all_phrases_ba: List[str] = []
        self.all_topics_ab: List[int] = []
        self.all_topics_ba: List[int] = []
        self._fitted = False

    def _partial_fit_one(
        self, model: BERTopic, phrases: List[str], all_phrases: List[str],
        all_topics: List[int], tag: str,
    ) -> None:
        if len(phrases) < self.n_topics:
            logging.warning(
                "BERTopic %s partition got %d new phrases (< n_topics=%d); "
                "partial_fit may fail or produce degenerate clusters",
                tag, len(phrases), self.n_topics,
            )
        if not phrases:
            return
        model.partial_fit(phrases)
        all_phrases.extend(phrases)
        all_topics.extend(model.topics_)
        model.topics_ = list(all_topics)

    def current_topic_metadata(
        self,
    ) -> Tuple[List[str], Dict[str, List[Tuple[str, int]]], Dict[str, int]]:
        """Return (feat_names, topic_phrases, topic_size) under current model state.

        Cheap: iterates the accumulated phrase/topic lists, no feature matrix build.
        """
        topic_ids_ab = sorted({t for t in self.all_topics_ab if t != -1})
        topic_ids_ba = sorted({t for t in self.all_topics_ba if t != -1})
        names: List[str] = []
        topic_phrases: Dict[str, List[Tuple[str, int]]] = {}
        topic_size: Dict[str, int] = {}
        for t in topic_ids_ab:
            name = f"[A>B] topic {t}"
            names.append(name)
            pc = _phrase_counts_for_topic(self.all_phrases_ab, self.all_topics_ab, t)
            topic_phrases[name] = pc
            topic_size[name] = sum(c for _, c in pc)
        for t in topic_ids_ba:
            name = f"[B>A] topic {t}"
            names.append(name)
            pc = _phrase_counts_for_topic(self.all_phrases_ba, self.all_topics_ba, t)
            topic_phrases[name] = pc
            topic_size[name] = sum(c for _, c in pc)
        return names, topic_phrases, topic_size

    def render_topics(
        self,
        selected_feat_names: List[str],
        top_k_phrases: int = 10,
        coefs: Dict[str, float] | None = None,
    ) -> str:
        """Convenience: render a list-of-lists for a subset of current topics."""
        _, topic_phrases, topic_size = self.current_topic_metadata()
        feats = SignedBertopicFeatures(
            X=np.zeros((0, 0), dtype=np.int64),
            feat_names=np.array([], dtype=object),
            topic_phrases=topic_phrases,
            topic_size=topic_size,
            model=self,
        )
        return feats.render_topics(selected_feat_names, top_k_phrases=top_k_phrases, coefs=coefs)

    def partial_fit(self, data_df: pd.DataFrame, text_summary_column: str) -> None:
        """Incorporate new comma-separated keyphrases from `data_df` into both models."""
        per_obs = _per_obs_phrases(data_df[text_summary_column])
        new_ab: List[str] = []
        new_ba: List[str] = []
        for obs_phrases in per_obs:
            for p in obs_phrases:
                sign = _parse_sign(p)
                if sign == "A>B":
                    new_ab.append(p)
                elif sign == "B>A":
                    new_ba.append(p)

        logging.info(
            "BERTopic partial_fit: +%d A>B phrases (%d total), +%d B>A phrases (%d total)",
            len(new_ab), len(self.all_phrases_ab) + len(new_ab),
            len(new_ba), len(self.all_phrases_ba) + len(new_ba),
        )
        self._partial_fit_one(
            self.model_ab, new_ab, self.all_phrases_ab, self.all_topics_ab, "A>B"
        )
        self._partial_fit_one(
            self.model_ba, new_ba, self.all_phrases_ba, self.all_topics_ba, "B>A"
        )
        self._fitted = True

    def build_features(
        self, data_df: pd.DataFrame, text_summary_column: str
    ) -> "SignedBertopicFeatures":
        """Build binary (n_obs, n_topics_total) features for `data_df`.

        Uses `transform()` to assign each phrase to a topic under the current
        model state (so works for both the training rows and new rows).
        """
        if not self._fitted:
            raise RuntimeError("SignedBertopicModel.build_features called before partial_fit")

        per_obs = _per_obs_phrases(data_df[text_summary_column])

        topic_ids_ab = sorted({t for t in self.all_topics_ab if t != -1})
        topic_ids_ba = sorted({t for t in self.all_topics_ba if t != -1})

        col_names: List[str] = []
        topic_phrases: Dict[str, List[Tuple[str, int]]] = {}
        topic_size: Dict[str, int] = {}

        for t in topic_ids_ab:
            name = f"[A>B] topic {t}"
            col_names.append(name)
            pc = _phrase_counts_for_topic(self.all_phrases_ab, self.all_topics_ab, t)
            topic_phrases[name] = pc
            topic_size[name] = sum(c for _, c in pc)
        for t in topic_ids_ba:
            name = f"[B>A] topic {t}"
            col_names.append(name)
            pc = _phrase_counts_for_topic(self.all_phrases_ba, self.all_topics_ba, t)
            topic_phrases[name] = pc
            topic_size[name] = sum(c for _, c in pc)

        n_cols_ab = len(topic_ids_ab)
        n_cols = n_cols_ab + len(topic_ids_ba)
        X = np.zeros((len(per_obs), n_cols), dtype=np.int64)

        if n_cols == 0:
            logging.warning("BERTopic produced zero non-noise topics")
            return SignedBertopicFeatures(
                X=X, feat_names=np.array(col_names, dtype=object),
                topic_phrases=topic_phrases, topic_size=topic_size, model=self,
            )

        topic_to_col_ab = {t: i for i, t in enumerate(topic_ids_ab)}
        topic_to_col_ba = {t: n_cols_ab + i for i, t in enumerate(topic_ids_ba)}

        for i, obs_phrases in enumerate(per_obs):
            ab = [p for p in obs_phrases if _parse_sign(p) == "A>B"]
            ba = [p for p in obs_phrases if _parse_sign(p) == "B>A"]
            if ab:
                topics_ab, _ = self.model_ab.transform(ab)
                for t in topics_ab:
                    if t in topic_to_col_ab:
                        X[i, topic_to_col_ab[t]] = 1
            if ba:
                topics_ba, _ = self.model_ba.transform(ba)
                for t in topics_ba:
                    if t in topic_to_col_ba:
                        X[i, topic_to_col_ba[t]] = 1

        logging.info(
            "BERTopic: %d A>B topics + %d B>A topics (noise dropped)",
            n_cols_ab, len(topic_ids_ba),
        )
        return SignedBertopicFeatures(
            X=X,
            feat_names=np.array(col_names, dtype=object),
            topic_phrases=topic_phrases,
            topic_size=topic_size,
            model=self,
        )


@dataclass
class SignedBertopicFeatures:
    """Container for sign-partitioned BERTopic topic features.

    X: (n_obs, n_topics_total) binary matrix.
    feat_names: short stable topic IDs ("[A>B] topic 3", "[B>A] topic 0").
    topic_phrases: feat_name -> list of (phrase, count) tuples sorted by count desc.
    topic_size: feat_name -> total count of phrases assigned to that topic.
    model: the underlying SignedBertopicModel; keep a reference so later MH
        iterations can call `.partial_fit(...)` and rebuild features.
    """

    X: np.ndarray
    feat_names: np.ndarray  # dtype=object
    topic_phrases: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)
    topic_size: Dict[str, int] = field(default_factory=dict)
    model: SignedBertopicModel | None = None

    def render_topics(
        self,
        selected_feat_names: List[str],
        top_k_phrases: int = 10,
        coefs: Dict[str, float] | None = None,
    ) -> str:
        """Render the given topics as a list-of-lists prompt block.

        Format per topic (coefs is signed, from a residual LR on Y):
            [A>B] topic 3 (n=42, coef=+0.73):
             - "more practical: A > B" (18)
             - "more actionable: A > B" (12)
             ...
        """
        blocks = []
        for name in selected_feat_names:
            n = self.topic_size.get(name, 0)
            phrases = self.topic_phrases.get(name, [])
            header = f"{name} (n={n}"
            if coefs is not None and name in coefs:
                header += f", coef={coefs[name]:+.3f}"
            header += "):"
            lines = [header]
            for phrase, count in phrases[:top_k_phrases]:
                lines.append(f'  - "{phrase}" ({count})')
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


def _phrase_counts_for_topic(
    phrases: List[str], phrase_topics: List[int], topic_id: int
) -> List[Tuple[str, int]]:
    """Return [(phrase, count), ...] sorted by count desc for a topic."""
    counter: Counter = Counter()
    for p, t in zip(phrases, phrase_topics):
        if t == topic_id:
            counter[p] += 1
    return counter.most_common()


def build_signed_bertopic_features(
    data_df: pd.DataFrame,
    text_summary_column: str,
    nr_topics: int = 20,
    seed: int = 0,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> SignedBertopicFeatures:
    """Convenience: build an online SignedBertopicModel, partial_fit once, emit features.

    For MH iterations, instead instantiate SignedBertopicModel directly and
    call `.partial_fit()` on each new batch; then `.build_features()` to get
    an updated feature matrix.
    """
    model = SignedBertopicModel(
        n_topics=nr_topics, seed=seed, embedding_model=embedding_model
    )
    model.partial_fit(data_df, text_summary_column)
    return model.build_features(data_df, text_summary_column)
