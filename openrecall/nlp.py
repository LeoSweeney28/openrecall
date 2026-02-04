import logging
from typing import Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384  # Dimension for all-MiniLM-L6-v2

model: Optional["SentenceTransformer"] = None


def _load_model() -> None:
    global model
    if model is not None:
        return
    if SentenceTransformer is None:
        logger.warning(
            "sentence-transformers is not installed. Embeddings will be zeros."
        )
        return
    try:
        model = SentenceTransformer(MODEL_NAME)
        logger.info(f"SentenceTransformer model '{MODEL_NAME}' loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model '{MODEL_NAME}': {e}")
        model = None


def get_embedding(text: str) -> np.ndarray:
    """
    Generates a sentence embedding for the given text.

    Splits the text into lines, encodes each line using the pre-loaded
    SentenceTransformer model, and returns the mean of the embeddings.
    Handles empty input text by returning a zero vector.

    Args:
        text: The input string to embed.

    Returns:
        A numpy array representing the mean embedding of the text lines,
        or a zero vector if the input is empty, whitespace only, or the
        model failed to load. The array type is float32.
    """
    _load_model()
    if model is None:
        logger.error("SentenceTransformer model is not available. Returning zero vector.")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    if not text or text.isspace():
        logger.warning("Input text is empty or whitespace. Returning zero vector.")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # Split text into non-empty lines
    sentences = [line for line in text.split("\n") if line.strip()]

    if not sentences:
        logger.warning("No non-empty lines found after splitting. Returning zero vector.")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    try:
        sentence_embeddings = model.encode(sentences)
        # Calculate the mean embedding
        mean_embedding = np.mean(sentence_embeddings, axis=0, dtype=np.float32)
        return mean_embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two numpy vectors.

    Args:
        a: The first numpy array.
        b: The second numpy array.

    Returns:
        The cosine similarity score (float between -1 and 1),
        or NaN if either vector has zero magnitude.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        logger.warning("One or both vectors have zero magnitude. Returning NaN.")
        return float("nan")

    similarity = np.dot(a, b) / (norm_a * norm_b)
    # Clip values to handle potential floating-point inaccuracies slightly outside [-1, 1]
    return float(np.clip(similarity, -1.0, 1.0))
