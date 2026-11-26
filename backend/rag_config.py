import re

TEXT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"

FALLBACK_TEXT_DENSE_DIM = 384

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


RERANK_MULTIPLIER = 4
RERANK_MAX_CANDIDATES = 100

CROSS_ENCODER_DEVICE = None
CROSS_ENCODER_BATCH_SIZE = 16

PAD_TO_MULTIPLE_OF = 64


def query_prefix_for(model_name):
    if not model_name:
        return "", ""
    n = model_name.lower()
    if "bge" in n and "-en" in n and "m3" not in n:
        return "Represent this sentence for searching relevant passages: ", ""
    if re.search(r"(^|[/_-])e5-", n):
        return "query: ", "passage: "
    return "", ""