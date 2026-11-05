"""t-SNE Similarity Matrix"""
import math

def compute_pairwise_affinity(dists_sq, beta=1.0):
    exps = [math.exp(-d * beta) for d in dists_sq]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

if __name__ == "__main__":
    print("[RUN] t-SNE Similarity Matrix...")
    aff = compute_pairwise_affinity([0.5, 1.2, 0.2])
    print(f"[OK] t-SNE Pairwise Affinities: {[round(p, 4) for p in aff]}")
