import argparse
import json
import logging
import os

import numpy as np
import strsimpy

# from nltk.metrics.distance import edit_distance, jaccard_distance, jaccard_index,
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from flexolmo.data_utils import load_data

log = logging.getLogger(__name__)

"""

This script loads a pre-trained model and tokenizer from the Hugging Face Transformers library.

Inputs: 
# Model: The HF model from which to extract training data.
# Prefix Length: The number of tokens to use as the prefix for generation.
# Number of Prefixes: The total number of prefixes to generate and use.
# Number of Samples per prefix: The number of samples to generate for each prefix.
# Generation Length: The number of tokens to generate for each sample.
# Source Data: The source data to consider as the training data, which will be used to sample prefixes and see if the generation matches with its continuation. (in .jsonl format)
# Matching Criteria Hyperparameters: Parameters for deciding whether a generated sample matches training data. This could involve thresholds based on edit distance, BM25 scores, or similar metrics.

Outputs:
# Generated Samples: The generated samples based on the prefixes and the model.
The code should print how many data points from the training data were successfully generated.
"""


def load_resources(args):
    source_data = load_data(
        args.domain, args.source_data, p=0.01, debug=False
    )  # debug=True to load a small sample for testing
    np.random.seed(2025)
    np.random.shuffle(source_data)

    num_prefixes = args.num_prefixes
    if num_prefixes > len(source_data):
        print(
            "WARNING: Number of prefixes exceeds the number of training data points. Adjusting to use all training data points."
        )
        num_prefixes = len(source_data)

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    return source_data, num_prefixes, model, tokenizer


def main(args):
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)

    avg_scores = []
    max_scores = []

    total_samples = 0
    similar_samples = 0

    ### measure similarity
    if args.matching_criteria == "levenshtein":
        levenshtein = strsimpy.levenshtein.Levenshtein()
    if args.matching_criteria == "normalized_levenshtein":
        normalized_levenshtein = strsimpy.normalized_levenshtein.NormalizedLevenshtein()
    if args.matching_criteria == "jaccard":
        jaccard = strsimpy.jaccard.Jaccard()
    if args.matching_criteria == "damerau":
        damerau = strsimpy.damerau.Damerau()
    if args.matching_criteria == "cosine":
        cosine = strsimpy.cosine.Cosine()

    def match_function(orig, decoded_samples):
        scores = []

        # Check if the generated sample matches the training data
        for decoded_sample in decoded_samples:
            if args.matching_criteria == "levenshtein":
                score = levenshtein.distance(orig, decoded_sample)
            elif args.matching_criteria == "normalized_levenshtein":
                score = normalized_levenshtein.similarity(orig, decoded_sample)
            elif args.matching_criteria == "jaccard":
                score = jaccard.similarity(orig, decoded_sample)
            elif args.matching_criteria == "damerau":
                score = damerau.distance(orig, decoded_sample)
            elif args.matching_criteria == "cosine":
                score = cosine.similarity(orig, decoded_sample)
            elif args.matching_criteria == "bm25":
                bm25 = BM25Okapi([orig.split()])
                score = bm25.get_scores(decoded_sample.split())[0]
            scores.append(score)

        does_match = False
        if args.matching_criteria in ["levenshtein", "damerau"]:
            # For distance metrics, lower scores indicate more similarity
            if np.any([score <= args.distance_threshold for score in scores]):
                does_match = True
        else:
            if np.any([score >= args.similarity_threshold for score in scores]):
                does_match = True

        avg_scores.append(np.mean(scores))
        max_scores.append(np.max(scores))

        return does_match

    assert args.prefix_length < args.min_generation_length <= args.generation_length
    save_file = f"{args.output_dir}/{args.num_prefixes}_{args.num_generations_per_prefix}_{args.prefix_length}_{args.generation_length}.jsonl"

    if False:  # os.path.exists(save_file):
        print("Start reading!")
        with open(save_file, "r") as f:
            for line in tqdm(f, total=args.num_prefixes):
                dp = json.loads(line)
                if match_function(dp["original"], dp["generated_samples"]):
                    similar_samples += 1
                total_samples += 1

    else:
        source_data, num_prefixes, model, tokenizer = load_resources(args)
        np.random.seed(2025)

        with open(save_file, "w") as f, tqdm(total=num_prefixes, smoothing=0) as progress:

            for i in tqdm(range(num_prefixes)):
                source = source_data[i]  # ['text']
                input_ids = tokenizer.encode(source, return_tensors="pt")
                if input_ids.shape[-1] < args.min_generation_length:
                    continue

                idx = np.random.randint(max(len(input_ids[0]) - args.generation_length, 1))

                prefix_input_ids = input_ids[:, idx : idx + args.prefix_length]
                prefix = tokenizer.decode(prefix_input_ids[0], skip_special_tokens=True)

                generation_length = min(args.generation_length, input_ids.shape[-1] - idx)
                orig_input_ids = input_ids[:, idx : idx + generation_length]
                assert orig_input_ids.shape[-1] == generation_length
                orig = tokenizer.decode(orig_input_ids[0], skip_special_tokens=True)

                generated_samples = model.generate(
                    prefix_input_ids.to(model.device),
                    max_length=generation_length,
                    do_sample=True,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    num_return_sequences=args.num_generations_per_prefix,
                )

                # Decode the generated samples
                decoded_samples = [
                    tokenizer.decode(s, skip_special_tokens=True) for s in generated_samples
                ]

                # Save the generated samples to a file
                if args.save_generated_samples:
                    f.write(
                        json.dumps(
                            {
                                "generated_samples": decoded_samples,
                                "prefix": prefix,
                                "original": orig,
                            }
                        )
                        + "\n"
                    )

                total_samples += 1
                if match_function(orig, decoded_samples):
                    similar_samples += 1

            progress.update(1)

    print(f"Number of total samples: {total_samples}")
    print(f"Number of highly similar samples: {similar_samples}")

    print(f"Avg. Scores: {np.mean(avg_scores)}")
    print(f"Avg. Max Scores: {np.mean(max_scores)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate samples from a pre-trained model and evaluate their similarity with given texts"
    )
    parser.add_argument(
        "--model_name", type=str, default="EleutherAI/pythia-70m", help="Name of the model to use"
    )
    parser.add_argument("--prefix_length", type=int, default=10, help="Length of the prefix to use")
    parser.add_argument(
        "--num_generations_per_prefix",
        type=int,
        default=1,
        help="Number of generations to make per prefix",
    )
    parser.add_argument(
        "--num_prefixes", type=int, default=5, help="Number of prefixes to generate from"
    )
    parser.add_argument(
        "--generation_length", type=int, default=256, help="Length of the generated samples"
    )
    parser.add_argument(
        "--min_generation_length", type=int, default=128, help="Length of the generated samples"
    )
    parser.add_argument("--domain", type=str, help="Domain name of the source data")
    parser.add_argument("--source_data", type=str, help="Path to the source data file")
    parser.add_argument(
        "--matching_criteria",
        type=str,
        default="levenshtein",
        choices=["levenshtein", "normalized_levenshtein", "jaccard", "damerau", "cosine", "bm25"],
        help="Criteria for matching generated samples with training data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Directory to save the generated samples"
    )
    parser.add_argument(
        "--save_generated_samples",
        action="store_true",
        help="Whether to save the generated samples to a file",
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Top-k sampling parameter for generation"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter for generation",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling during generation"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
        help="Threshold for similarity score to consider a match",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=10,
        help="Threshold for distance score to consider a match",
    )

    args = parser.parse_args()
    main(args)
