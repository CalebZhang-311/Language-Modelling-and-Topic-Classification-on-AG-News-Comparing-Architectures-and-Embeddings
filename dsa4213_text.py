"""Project-specific text preparation for the DSA4213 AG News experiments."""

from collections import Counter
import html
import re
import unicodedata

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
TOKEN_PATTERN = re.compile(r"\w+(?:'\w+)*|[^\w\s]", flags=re.UNICODE)


def clean_text(text):
    """Normalize typography, HTML formatting, and whitespace consistently."""
    text = unicodedata.normalize("NFKC", text)
    # AG News includes malformed numeric entities such as '#36;'.
    text = re.sub(r"(?<!&)#(\d{1,7});", r"&#\1;", text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(str.maketrans({
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
    }))
    return " ".join(text.lower().split())


def tokenize(text):
    """Split cleaned text into words (including contractions) and punctuation."""
    return TOKEN_PATTERN.findall(clean_text(text))


def build_vocabulary(training_texts, max_size=10_000, min_frequency=2):
    """Fit a deterministic vocabulary using training text only."""
    if max_size < len(SPECIAL_TOKENS):
        raise ValueError("max_size must accommodate all special tokens")
    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")
    counts = Counter()
    for text in training_texts:
        counts.update(tokenize(text))
    candidates = sorted(
        (token for token, count in counts.items()
         if count >= min_frequency and token not in SPECIAL_TOKENS),
        key=lambda token: (-counts[token], token),
    )
    id_to_token = list(SPECIAL_TOKENS) + candidates[:max_size - len(SPECIAL_TOKENS)]
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    return token_to_id, id_to_token, counts


def encode_text(text, token_to_id):
    """Encode one news item, keeping explicit beginning and end boundaries."""
    ids = [token_to_id.get(token, token_to_id["<unk>"]) for token in tokenize(text)]
    return [token_to_id["<bos>"]] + ids + [token_to_id["<eos>"]]


def decode_ids(ids, id_to_token):
    """Return tokens for inspecting an encoded example."""
    return [id_to_token[index] for index in ids]
