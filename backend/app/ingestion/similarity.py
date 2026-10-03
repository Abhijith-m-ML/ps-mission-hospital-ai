import math
from typing import Sequence, Union
import numpy as np


def cosine_similarity(
    vec_a: Union[Sequence[float], np.ndarray],
    vec_b: Union[Sequence[float], np.ndarray],
) -> float:
    """Calculates the cosine similarity between two numerical vectors.
    
    Formula:
        cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
        
    Properties:
        - Output is in [-1.0, 1.0].
        - Identical directional vectors produce 1.0.
        - Orthogonal (perpendicular) vectors produce 0.0.
        - Exactly opposing vectors produce -1.0.
        - Zero vectors return 0.0 gracefully without division by zero.
    
    Args:
        vec_a: First vector (list, tuple, or 1D numpy array).
        vec_b: Second vector (list, tuple, or 1D numpy array).
        
    Returns:
        float: Cosine similarity score between -1.0 and 1.0.
        
    Raises:
        ValueError: If vectors have different dimensions or are empty.
    """
    if len(vec_a) == 0 or len(vec_b) == 0:
        raise ValueError("Cannot calculate cosine similarity for empty vectors.")

    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vectors must have the same dimension. Received dimensions: {len(vec_a)} and {len(vec_b)}."
        )

    # Use pure math / numpy-compatible iteration for robustness
    dot_product = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    for a, b in zip(vec_a, vec_b):
        val_a = float(a)
        val_b = float(b)
        dot_product += val_a * val_b
        norm_a_sq += val_a * val_a
        norm_b_sq += val_b * val_b

    norm_a = math.sqrt(norm_a_sq)
    norm_b = math.sqrt(norm_b_sq)

    # Zero-vector safety: if either vector has zero magnitude, similarity is 0.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)
    # Clip float inaccuracies to [-1.0, 1.0]
    return max(-1.0, min(1.0, float(similarity)))
