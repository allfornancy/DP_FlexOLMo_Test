import argparse
import logging
import random
from pathlib import Path

import numpy as np
from gritlm import GritLM
from sentence_transformers import SentenceTransformer

from flexolmo.data_utils import load_data

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Get domain embeddings")
    parser.add_argument(
        "-m",
        "--embedding-model",
        type=str,
        default="grit",
        choices=["grit", "mistral", "croissant"],
        help="Model to use for embeddings",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to domain data files; glob patterns are supported.",
    )
    parser.add_argument(
        "-d",
        "--domain-name",
        type=str,
        help="Domain name",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="Path to save the embeddings",
    )
    return parser.parse_args()


def load_embedding_model(model_name: str):
    if "grit" in model_name:
        model = GritLM("GritLM/GritLM-7B", torch_dtype="auto", mode="embedding", device_map="auto")
    elif "mistral" in model_name:
        model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")
    elif "croissant" in model_name:
        model = SentenceTransformer("manu/sentence_croissant_alpha_v0.4")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model


def get_embeddings(domain: str, data_path: str, model_name: str, output_path: str):
    model = load_embedding_model(model_name)
    docs = load_data(domain, data_path, p=0.0001, debug=True)
    n_tokens = [len(doc.split()) for doc in docs]
    print("number of docs: ", len(docs))
    print("number of tokens: ", np.mean(n_tokens))
    random.shuffle(docs)
    subsample_docs = docs[:1000]
    d_rep = model.encode(subsample_docs, batch_size=32)  # instruction=gritlm_instruction(""),
    d_rep_drop_nan = d_rep[~np.isnan(d_rep).any(axis=1)]
    # take the mean of the embeddings
    d_rep = np.mean(d_rep_drop_nan, axis=0)
    # save the embeddings
    save_path = f"{output_path}/{model_name}/{domain}.npy"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    np.save(save_path, d_rep)
    logger.info(f"Embeddings for domain {domain} saved to {save_path}")


if __name__ == "__main__":

    args = parse_args()

    get_embeddings(args.domain_name, args.data_path, args.embedding_model, args.output_path)
