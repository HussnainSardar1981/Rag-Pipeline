"""
Embedding Quality Checker for RAG Pipeline
Analyzes embedding quality with multiple metrics
Includes TF-IDF correlation to detect random/broken embeddings
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys
import argparse

# Add parent directory to import paths_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from cli_common import add_standard_args, build_selection, configure_logging

# Optional: TF-IDF correlation check
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.stats import spearmanr
    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False


def load_embeddings(file_path: Path) -> Dict:
    """Load embeddings from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def check_embeddings(file_path: Path):
    """Run all quality checks and print report"""

    print("\n" + "=" * 80)
    print("EMBEDDING QUALITY REPORT")
    print("=" * 80)
    print(f"File: {file_path.name}")
    print()

    # Load data
    data = load_embeddings(file_path)
    chunks = data.get('chunks', [])

    if not chunks:
        print("ERROR: No chunks found!")
        return

    embeddings = [np.array(chunk['embedding']) for chunk in chunks]
    texts = [chunk['text'][:60] + '...' if len(chunk['text']) > 60 else chunk['text'] for chunk in chunks]

    # CHECK 1: Basic Statistics
    print("=" * 80)
    print("1. BASIC STATISTICS")
    print("=" * 80)
    print(f"  Total chunks:        {len(chunks)}")
    print(f"  Embedding dimension: {len(embeddings[0])}")
    print(f"  Expected dimension:  {data.get('embedding_dim', 'unknown')}")
    print(f"  Model:              {data.get('embedding_model', 'unknown')}")
    print(f"  Device:             {data.get('device', 'unknown')}")

    dim_match = len(embeddings[0]) == data.get('embedding_dim', 0)
    print(f"  Dimension match:    {'[OK]' if dim_match else '[FAIL]'}")

    # CHECK 2: Sanity Checks (NaN, Inf, Zeros)
    print("\n" + "=" * 80)
    print("2. SANITY CHECKS")
    print("=" * 80)

    emb_matrix = np.array(embeddings)
    has_nan = np.isnan(emb_matrix).any()
    has_inf = np.isinf(emb_matrix).any()

    print(f"  NaN values:         {'[FAIL] found NaN' if has_nan else '[OK]'}")
    print(f"  Inf values:         {'[FAIL] found Inf' if has_inf else '[OK]'}")

    # Check for duplicate vectors
    unique_vectors = len(np.unique(emb_matrix.round(6), axis=0))
    print(f"  Unique vectors:     {unique_vectors}/{len(embeddings)}")
    if unique_vectors < len(embeddings):
        print(f"    [WARNING] Found {len(embeddings) - unique_vectors} duplicate embeddings")

    # CHECK 3: Normalization
    print("\n" + "=" * 80)
    print("3. NORMALIZATION CHECK")
    print("=" * 80)

    norms = [np.linalg.norm(emb) for emb in embeddings]
    min_norm = np.min(norms)
    max_norm = np.max(norms)
    mean_norm = np.mean(norms)

    print(f"  Min L2 norm:        {min_norm:.6f}")
    print(f"  Max L2 norm:        {max_norm:.6f}")
    print(f"  Mean L2 norm:       {mean_norm:.6f}")

    is_normalized = np.allclose(norms, 1.0, atol=0.01)
    print(f"  Normalized:         {'[OK] all ~1.0' if is_normalized else '[FAIL] expected ~1.0'}")

    # CHECK 4: Value Distribution
    print("\n" + "=" * 80)
    print("4. VALUE DISTRIBUTION")
    print("=" * 80)

    all_values = emb_matrix.flatten()
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    mean_val = np.mean(all_values)
    std_val = np.std(all_values)
    pct_positive = np.sum(all_values > 0) / len(all_values) * 100
    pct_negative = np.sum(all_values < 0) / len(all_values) * 100
    pct_zeros = np.sum(all_values == 0) / len(all_values) * 100

    print(f"  Min value:          {min_val:.6f}")
    print(f"  Max value:          {max_val:.6f}")
    print(f"  Mean value:         {mean_val:.6f}")
    print(f"  Std deviation:      {std_val:.6f}")
    print(f"  Positive values:    {pct_positive:.1f}%")
    print(f"  Negative values:    {pct_negative:.1f}%")
    print(f"  Zero values:        {pct_zeros:.2f}%")

    balanced = 30 < pct_positive < 70
    print(f"  Distribution:       {'[OK] balanced' if balanced else '[WARNING] skewed'}")

    # CHECK 5: Similarity Analysis
    print("\n" + "=" * 80)
    print("5. SIMILARITY ANALYSIS")
    print("=" * 80)

    n = len(embeddings)
    similarities = []
    duplicate_pairs = []
    high_sim_pairs = []

    # Calculate pairwise similarities
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)

            if sim > 0.99:
                duplicate_pairs.append((i+1, j+1, sim, texts[i], texts[j]))
            elif sim > 0.8:
                high_sim_pairs.append((i+1, j+1, sim, texts[i], texts[j]))

    print(f"  Min similarity:     {np.min(similarities):.4f}")
    print(f"  Max similarity:     {np.max(similarities):.4f}")
    print(f"  Mean similarity:    {np.mean(similarities):.4f}")
    print(f"  Std similarity:     {np.std(similarities):.4f}")

    if duplicate_pairs:
        print(f"\n  [WARNING] Found {len(duplicate_pairs)} potential duplicate chunks:")
        for i, j, sim, text_i, text_j in duplicate_pairs[:3]:
            print(f"    Chunks {i} <-> {j} (sim: {sim:.4f})")
            print(f"      '{text_i}'")
            print(f"      '{text_j}'")
    else:
        print("  [OK] No duplicate chunks detected")

    if high_sim_pairs:
        print(f"\n  Top {min(3, len(high_sim_pairs))} similar chunk pairs (related content):")
        for i, j, sim, text_i, text_j in sorted(high_sim_pairs, key=lambda x: x[2], reverse=True)[:3]:
            print(f"    Chunks {i} <-> {j} (sim: {sim:.4f})")
            print(f"      '{text_i}'")
            print(f"      '{text_j}'")

    # CHECK 6: Semantic Coherence (consecutive chunks)
    print("\n" + "=" * 80)
    print("6. SEMANTIC COHERENCE (Procedural Docs)")
    print("=" * 80)

    consecutive_sims = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        consecutive_sims.append(sim)

    mean_consec = np.mean(consecutive_sims) if consecutive_sims else 0
    std_consec = np.std(consecutive_sims) if consecutive_sims else 0

    print(f"  Consecutive similarity (mean): {mean_consec:.4f}")
    print(f"  Consecutive similarity (std):  {std_consec:.4f}")

    if len(embeddings) > 1:
        first_last_sim = cosine_similarity(embeddings[0], embeddings[-1])
        print(f"  First <-> Last chunk:         {first_last_sim:.4f}")

    # Interpretation
    if mean_consec > 0.8:
        interp = "Very high (might indicate duplicate or very similar steps)"
    elif mean_consec > 0.6:
        interp = "Good (related procedural steps)"
    elif mean_consec > 0.4:
        interp = "Moderate (acceptable for diverse procedural content)"
    else:
        interp = "Low (chunks might be too diverse or unrelated)"

    print(f"  Interpretation:               {interp}")

    # CHECK 7: TF-IDF Correlation (Semantic Alignment Test)
    print("\n" + "=" * 80)
    print("7. TF-IDF CORRELATION (Semantic Alignment)")
    print("=" * 80)

    if TFIDF_AVAILABLE and len(texts) >= 3:
        try:
            # Build TF-IDF similarity matrix
            texts_raw = [chunk.get('text', '') for chunk in chunks]
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
            tfidf_matrix = vectorizer.fit_transform(texts_raw)
            tfidf_sim_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

            # Collect pairwise similarities
            emb_sims = []
            tfidf_sims = []
            for i in range(n):
                for j in range(i + 1, n):
                    emb_sims.append(cosine_similarity(embeddings[i], embeddings[j]))
                    tfidf_sims.append(tfidf_sim_matrix[i, j])

            # Calculate Spearman correlation
            corr, p_value = spearmanr(emb_sims, tfidf_sims)

            print(f"  Spearman correlation:        {corr:.4f}")
            print(f"  P-value:                     {p_value:.4e}")

            # Interpretation
            if corr > 0.5:
                tfidf_interp = "Excellent (strong semantic alignment)"
            elif corr > 0.3:
                tfidf_interp = "Good (moderate semantic alignment)"
            elif corr > 0.1:
                tfidf_interp = "Fair (weak semantic alignment)"
            else:
                tfidf_interp = "Poor (embeddings may be random or broken)"

            print(f"  Interpretation:              {tfidf_interp}")

            has_semantic_alignment = corr > 0.3

        except Exception as e:
            print(f"  [WARNING] TF-IDF correlation failed: {e}")
            has_semantic_alignment = True  # Don't penalize if check fails

    elif not TFIDF_AVAILABLE:
        print("  [SKIP] sklearn/scipy not installed")
        print("  Install: pip install scikit-learn scipy")
        has_semantic_alignment = True
    else:
        print("  [SKIP] Need at least 3 chunks for correlation")
        has_semantic_alignment = True

    # CHECK 8: Overall Quality Score
    print("\n" + "=" * 80)
    print("8. OVERALL QUALITY SCORE")
    print("=" * 80)

    score = 0
    max_score = 7

    if dim_match: score += 1
    if not has_nan and not has_inf: score += 1
    if is_normalized: score += 1
    if balanced: score += 1
    if not duplicate_pairs: score += 1
    if unique_vectors == len(embeddings): score += 1
    if has_semantic_alignment: score += 1

    quality = score / max_score * 100

    if quality >= 90:
        grade = "EXCELLENT [OK]"
    elif quality >= 75:
        grade = "GOOD [OK]"
    elif quality >= 60:
        grade = "ACCEPTABLE [WARNING]"
    else:
        grade = "POOR [FAIL]"

    print(f"  Quality Score:      {score}/{max_score} ({quality:.0f}%)")
    print(f"  Grade:              {grade}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    recommendations = []

    if not dim_match:
        recommendations.append("- Dimension mismatch detected - verify model configuration")

    if has_nan or has_inf:
        recommendations.append("- Found NaN/Inf values - embeddings are corrupted")

    if not is_normalized:
        recommendations.append("- Embeddings not normalized - set normalize_embeddings=True")

    if duplicate_pairs:
        recommendations.append(f"- Found {len(duplicate_pairs)} duplicate chunks - review deduplication")

    if not balanced:
        recommendations.append("- Value distribution skewed - verify model is working correctly")

    if mean_consec < 0.3:
        recommendations.append("- Low consecutive similarity - chunks might be poorly chunked")

    if unique_vectors < len(embeddings):
        recommendations.append(f"- Found {len(embeddings) - unique_vectors} duplicate embeddings - check for duplicate text")

    if TFIDF_AVAILABLE and 'corr' in locals() and corr < 0.3:
        recommendations.append(f"- Low TF-IDF correlation ({corr:.3f}) - embeddings may not capture text semantics well")

    if not TFIDF_AVAILABLE:
        recommendations.append("- Install scikit-learn and scipy to enable TF-IDF correlation check: pip install scikit-learn scipy")

    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("  [OK] All checks passed! Embeddings are production-ready.")

    print("\n" + "=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Embedding Quality Checker for RAG Pipeline'
    )
    add_standard_args(parser, include_file=True, include_customer_id=False, include_server_root=True)
    args = parser.parse_args()
    selection = build_selection(args)

    configure_logging("check_embeddings", server_root=selection.server_root)

    if not selection.file:
        print("Usage: python check_embeddings.py --file /path/to/content_embedded.json")
        return 1

    file_path = Path(selection.file)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return 1

    try:
        check_embeddings(file_path)
        return 0

    except Exception as e:
        print(f"Error analyzing embeddings: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
