import math
import pytest
import numpy as np
from app.ingestion.similarity import cosine_similarity


def test_identical_vectors():
    """Identical vectors should have a cosine similarity of exactly or virtually 1.0."""
    vec_a = [1.0, 2.0, 3.0]
    vec_b = [1.0, 2.0, 3.0]
    score = cosine_similarity(vec_a, vec_b)
    assert pytest.approx(score, rel=1e-5) == 1.0

    # Scaled identical vectors (same direction, different magnitude)
    vec_c = [2.0, 4.0, 6.0]
    score_scaled = cosine_similarity(vec_a, vec_c)
    assert pytest.approx(score_scaled, rel=1e-5) == 1.0


def test_orthogonal_vectors():
    """Orthogonal (perpendicular) vectors should have a cosine similarity of 0.0."""
    vec_a = [1.0, 0.0]
    vec_b = [0.0, 1.0]
    score = cosine_similarity(vec_a, vec_b)
    assert pytest.approx(score, abs=1e-6) == 0.0

    vec_3d_1 = [1.0, 0.0, 0.0]
    vec_3d_2 = [0.0, 0.0, 5.0]
    assert pytest.approx(cosine_similarity(vec_3d_1, vec_3d_2), abs=1e-6) == 0.0


def test_opposite_vectors():
    """Opposite vectors should have a cosine similarity of -1.0."""
    vec_a = [1.0, 2.0, 3.0]
    vec_b = [-1.0, -2.0, -3.0]
    score = cosine_similarity(vec_a, vec_b)
    assert pytest.approx(score, rel=1e-5) == -1.0


def test_similar_vectors():
    """Similar directional vectors should have a high positive cosine similarity."""
    vec_a = [1.0, 2.0, 3.0]
    vec_b = [1.1, 1.9, 3.2]
    score = cosine_similarity(vec_a, vec_b)
    assert 0.95 <= score <= 1.0


def test_different_vectors():
    """Uncorrelated vectors have lower similarity than similar vectors."""
    vec_a = [1.0, 2.0, 0.0]
    vec_similar = [1.1, 2.1, 0.1]
    vec_different = [-1.0, 0.5, 3.0]

    sim_high = cosine_similarity(vec_a, vec_similar)
    sim_low = cosine_similarity(vec_a, vec_different)
    assert sim_high > sim_low


def test_zero_vector_handling():
    """Zero vectors should return 0.0 without division by zero errors."""
    zero_vec = [0.0, 0.0, 0.0]
    normal_vec = [1.0, 2.0, 3.0]

    assert cosine_similarity(zero_vec, normal_vec) == 0.0
    assert cosine_similarity(normal_vec, zero_vec) == 0.0
    assert cosine_similarity(zero_vec, zero_vec) == 0.0


def test_numpy_array_support():
    """Cosine similarity function accepts numpy arrays seamlessly."""
    arr_a = np.array([0.5, 0.5, 0.5])
    arr_b = np.array([0.5, 0.5, 0.5])
    score = cosine_similarity(arr_a, arr_b)
    assert pytest.approx(score, rel=1e-5) == 1.0


def test_dimension_mismatch_error():
    """Vectors of different lengths must raise ValueError."""
    vec_a = [1.0, 2.0]
    vec_b = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="same dimension"):
        cosine_similarity(vec_a, vec_b)


def test_empty_vector_error():
    """Empty vectors must raise ValueError."""
    with pytest.raises(ValueError, match="empty vectors"):
        cosine_similarity([], [])
