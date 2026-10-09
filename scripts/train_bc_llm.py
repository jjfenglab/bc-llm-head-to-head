"""
End-to-end BC-LLM trainer: proposes an initial concept set from top word
features (what train_baseline.py did), then runs Bayesian MH iterations
over the concept set (what train_bayesian.py did).

Collapses the previous two-step pipeline into a single entry point. CLI
flags are the union of the two originals; shared helpers are reused via
imports rather than copied.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

sys.path.append(os.getcwd())
from src.concept_learner_model import ConceptLearnerModel, CandidateConcepts
from src.llm_utils import LLMApi, create_llm_clients
from src.training_history import TrainingHistory
import src.common as common
from scripts.train_bayesian import load_data_partition, get_word_count_data, load_llms
from scripts.train_baseline import generate_prior_prompt


def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__)
    # Data
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--in-dataset-file", type=str, required=True,
                        help="csv of the labelled training data")
    parser.add_argument("--indices-csv", type=str, required=True,
                        help="csv of training indices (partition column)")
    parser.add_argument("--init-concepts-file", type=str, default=None,
                        help="optional precomputed LLM-extracted keyphrases to seed llm_output")
    parser.add_argument("--text-summary-column", type=str, default="llm_output",
                        choices=['llm_output', 'sentence', 'spacy_output'])
    parser.add_argument("--keep-x-cols", type=str, nargs="*",
                        help="tabular columns to force keep")
    parser.add_argument("--max-obs", type=int, default=0)
    parser.add_argument("--min-prevalence", type=float, default=0)
    parser.add_argument("--cluster-column", type=str, default=None,
                        help="Column whose values define cluster IDs; rows sharing a "
                             "value are kept in the same partial-posterior split "
                             "(needed for augmented symmetric pairs).")
    # LLM
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--cache-file", type=str, default="cache.db")
    parser.add_argument("--llm-model-type", type=str)
    parser.add_argument("--llm-iter-type", type=str)
    parser.add_argument("--llm-extraction-type", type=str)
    # Baseline-init step
    parser.add_argument("--baseline-init-file", type=str, required=True,
                        help="prompt template for generating the initial concept set")
    parser.add_argument("--num-top-attributes", type=int, default=40,
                        help="number of top words to show the LLM when proposing init concepts")
    parser.add_argument("--use-acc", action="store_true",
                        help="rank top words by accuracy rather than AUC")
    parser.add_argument("--max-tokens", type=int, default=10000,
                        help="max_new_tokens for the init-concepts LLM call")
    parser.add_argument("--config-file", type=str,
                        help="json of substitutions to apply to the init prompt")
    # Bayesian MH step
    parser.add_argument("--num-meta-concepts", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--num-greedy-epochs", type=int, default=0)
    parser.add_argument("--num-greedy-holdout", type=int, default=1)
    parser.add_argument("--is-greedy-metric-acc", action="store_true", default=False)
    parser.add_argument("--do-greedy", action="store_true", default=False)
    parser.add_argument("--num-minibatch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-concept-size", type=int, default=20)
    parser.add_argument("--batch-obs-size", type=int, default=1)
    parser.add_argument("--learner-type", type=str, default="count_l2",
                        choices=['count_elasticnet', 'tfidf_l2', 'tfidf_l1',
                                 'count_l2', 'count_l1', 'count_rf'])
    parser.add_argument("--final-learner-type", type=str, default=None,
                        choices=['l1', 'l2', None])
    parser.add_argument("--inverse-penalty-param", type=int, default=20000)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--prompt-iter-file", type=str, required=True)
    parser.add_argument("--prompt-iter-type", type=str, default="conditional",
                        choices=["marginal", "conditional", "error_obs", "error_obs_w_conditional"])
    parser.add_argument("--prompt-concepts-file", type=str, required=True)
    parser.add_argument("--prompt-prior-file", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument("--num-top-residual-words", type=int, default=40)
    parser.add_argument("--max-section-length", type=int, default=None)
    parser.add_argument("--is-image", action="store_true", default=False)
    # Outputs
    parser.add_argument("--log-file", type=str, default="_output/log_train_bc_llm.txt")
    parser.add_argument("--training-history-file", type=str, required=True)
    parser.add_argument("--out-extractions-file", type=str,
                        default="_output/note_extractions.pkl")
    parser.add_argument("--aucs-plot-file", type=str, default="_output/aucs.png")

    args = parser.parse_args()
    assert args.use_api, "--use-api required (vLLM path not supported in this shim)"
    args.partition = "train"
    args.count_vectorizer, args.residual_model_type = args.learner_type.split("_")
    # Baseline helper reads args.model (its own learner-type parse); mirror it.
    args.model = args.residual_model_type if args.residual_model_type != "None" else None
    args.keep_x_cols = pd.Series(args.keep_x_cols) if args.keep_x_cols is not None else None
    return args


def propose_initial_concepts(args, data_df, X_words, y, word_names, llm_dict) -> list[dict]:
    """Baseline step: ask the LLM for an initial concept set using top-word prompts."""
    init_prompt = generate_prior_prompt(
        data_df,
        X_words,
        y,
        word_names=word_names,
        args=args,
        num_top=args.num_top_attributes,
        use_acc=args.use_acc,
    )
    logging.info("INIT LLM PROMPT %s", init_prompt)
    candidate = llm_dict["iter"].get_output(
        init_prompt,
        max_new_tokens=args.max_tokens,
        response_model=CandidateConcepts,
    )
    logging.info("INIT LLM RESPONSE %s", candidate)
    return candidate.to_dicts()[: args.num_meta_concepts]


def main(args):
    args = parse_args(args)
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    load_dotenv()

    # Load the training partition and word-level features (shared with both legacy scripts).
    data_df = load_data_partition(
        args,
        init_concepts_file=args.init_concepts_file,
        text_summary_column=args.text_summary_column,
    )
    X_features, feat_names = get_word_count_data(
        data_df,
        args.count_vectorizer,
        args.text_summary_column,
        min_prevalence=args.min_prevalence,
    )
    y_train = data_df["y"].to_numpy().flatten()
    logging.info("data_df shape %s, X_features shape %s, num classes %d",
                 data_df.shape, X_features.shape, np.unique(y_train).size)

    llm_dict = load_llms(args)

    # Step 1: propose initial concepts from the top-word residual model.
    init_concepts = propose_initial_concepts(args, data_df, X_features, y_train, feat_names, llm_dict)
    logging.info("initial concepts: %s", [c["concept"] for c in init_concepts])

    history = TrainingHistory(args.keep_x_cols)
    history.add_concepts(init_concepts)

    # Step 2: MH iterations over the concept set, seeded with those initial concepts.
    bayesian_cbm = ConceptLearnerModel(
        init_history=history,
        llm_iter=llm_dict["iter"],
        llm_extraction=llm_dict["extraction"],
        num_classes=np.unique(y_train).size,
        num_minibatch=args.num_minibatch,
        num_meta_concepts=args.num_meta_concepts,
        prompt_iter_type=args.prompt_iter_type,
        prompt_iter_file=args.prompt_iter_file,
        config={},
        prompt_concepts_file=args.prompt_concepts_file,
        prompt_prior_file=args.prompt_prior_file,
        out_extractions_file=args.out_extractions_file,
        residual_model_type=args.residual_model_type,
        final_learner_type=args.final_learner_type,
        inverse_penalty_param=args.inverse_penalty_param,
        train_frac=args.train_frac,
        num_greedy_epochs=args.num_greedy_epochs,
        num_greedy_holdout=args.num_greedy_holdout,
        is_greedy_metric_acc=args.is_greedy_metric_acc,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        batch_concept_size=args.batch_concept_size,
        batch_obs_size=args.batch_obs_size,
        do_greedy=args.do_greedy,
        is_image=args.is_image,
        all_extracted_features_dict={},
        max_section_length=args.max_section_length,
        force_keep_columns=args.keep_x_cols,
        num_top=args.num_top_residual_words,
        max_new_tokens=args.max_new_tokens,
        cluster_column=args.cluster_column,
    )
    bayesian_cbm.fit(
        data_df,
        X_features=X_features,
        feat_names=feat_names,
        training_history_file=args.training_history_file,
        aucs_plot_file=args.aucs_plot_file,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
