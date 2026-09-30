"""
First-order autoregressive language model (Bayesian network X1 -> X2 -> ... -> XT)

    P(X1..XT) = P(X1 | <START>) * prod_{t=2..T} P(Xt | Xt-1)

The model is built purely from transition counts using ordinary Python data
structures and random sampling. No ML library, no pretrained model.

Run:  python first_order.py            (optionally:  python first_order.py 123   -> seed)
"""

import os
import random
import sys
from collections import Counter, defaultdict

START = "<START>"
END = "<END>"

DATASET = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]

MAX_LEN = 20  # safety cap: greedy decoding can loop forever (see Question 10)


# ----------------------------------------------------------------------------
# Tokenisation helpers
# ----------------------------------------------------------------------------
def tokenise(sentence):
    """Lower-case, split on whitespace, add <START> and <END>."""
    return [START] + sentence.lower().split() + [END]


def detokenise(tokens):
    """Join tokens back into a sentence, dropping the special tokens."""
    return " ".join(t for t in tokens if t not in (START, END))


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------
class FirstOrderModel:
    """P(Xt | Xt-1) estimated from bigram counts."""

    def __init__(self):
        # counts[prev][next] = C(prev, next)   <-- transition counts live here
        self.counts = defaultdict(Counter)
        # probabilities[prev][next] = P(next | prev)   <-- the CPT
        self.probabilities = {}

    # -- learning ------------------------------------------------------------
    def train(self, tokenised_sentences):
        for tokens in tokenised_sentences:
            for prev, nxt in zip(tokens[:-1], tokens[1:]):
                self.counts[prev][nxt] += 1
        self._build_probabilities()

    def _build_probabilities(self):
        """P(wj | wi) = C(wi, wj) / sum_k C(wi, wk)"""
        self.probabilities = {}
        for prev, next_counts in self.counts.items():
            total = sum(next_counts.values())
            dist = {}
            for nxt, count in next_counts.items():
                dist[nxt] = count / total
            self.probabilities[prev] = dist

    # -- inspection ----------------------------------------------------------
    def distribution(self, prev):
        """Return {next: prob} for `prev`, or None if `prev` was never seen."""
        return self.probabilities.get(prev)

    def show_distribution(self, prev):
        dist = self.distribution(prev)
        if dist is None:
            print("  P(next | %s): no observed transitions" % prev)
            return
        print("  P(next | %s):" % prev)
        for nxt in sorted(dist, key=dist.get, reverse=True):
            print("      %-8s %.4f" % (nxt, dist[nxt]))

    def vocabulary(self):
        """All tokens that can appear as a *next* token (words + <END>)."""
        vocab = set()
        for next_counts in self.counts.values():
            vocab.update(next_counts.keys())
        return vocab

    def zero_transitions(self, prev):
        """Tokens that never follow `prev` in the training data."""
        dist = self.distribution(prev) or {}
        return sorted(self.vocabulary() - set(dist.keys()))

    def num_parameters(self):
        """Number of non-zero CPT entries (distinct estimated probabilities)."""
        return sum(len(d) for d in self.probabilities.values())

    # -- prediction ----------------------------------------------------------
    def most_probable(self, prev):
        """arg max_w P(w | prev). Ties broken alphabetically for determinism."""
        dist = self.distribution(prev)
        if dist is None:
            return None
        best_word = None
        best_prob = -1.0
        for word in sorted(dist):
            if dist[word] > best_prob:
                best_word = word
                best_prob = dist[word]
        return best_word

    def sample_next(self, prev):
        """Draw w ~ P(w | prev)."""
        dist = self.distribution(prev)
        if dist is None:
            return None
        words = list(dist.keys())
        weights = list(dist.values())
        return random.choices(words, weights=weights, k=1)[0]

    # -- generation ----------------------------------------------------------
    def generate(self, mode="sample", max_len=MAX_LEN):
        """
        mode = "sample": X_t ~ P(. | X_{t-1})
        mode = "greedy": X_t = arg max P(. | X_{t-1})
        Stops at <END>, at an unseen context, or after max_len words.
        """
        words = []
        prev = START
        while len(words) < max_len:
            if mode == "greedy":
                nxt = self.most_probable(prev)
            else:
                nxt = self.sample_next(prev)
            if nxt is None or nxt == END:  # unseen context or end of sentence
                break
            words.append(nxt)
            prev = nxt
        return " ".join(words)

    # -- testing -------------------------------------------------------------
    def check_normalisation(self, tolerance=1e-9, verbose=True):
        """For every context w:  sum_v P(v | w) must equal 1."""
        all_ok = True
        for prev in sorted(self.probabilities):
            total = sum(self.probabilities[prev].values())
            ok = abs(total - 1.0) < tolerance
            all_ok = all_ok and ok
            if verbose:
                print("  %-8s sum = %.6f  %s" % (prev, total, "OK" if ok else "FAIL"))
        return all_ok


# ----------------------------------------------------------------------------
# Demo / lab driver
# ----------------------------------------------------------------------------
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    random.seed(seed)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    data = [tokenise(s) for s in DATASET]
    model = FirstOrderModel()
    model.train(data)

    print("=" * 60)
    print("PART IV / Q3: CPTs  P(next | current)")
    print("=" * 60)
    for word in [START, "the", "cat", "dog", "sat", "ran"]:
        model.show_distribution(word)
        print("      zero-probability next tokens:", model.zero_transitions(word))

    print()
    print("=" * 60)
    print("PART VII / Q8: normalisation test")
    print("=" * 60)
    print("All contexts sum to 1:", model.check_normalisation())

    # Demonstrate that the test catches a bug (a transition silently dropped)
    print("\nDeliberately broken copy (one transition removed, NOT renormalised):")
    broken = FirstOrderModel()
    broken.train(data)
    del broken.probabilities["the"]["park"]
    print("  'the' total =", round(sum(broken.probabilities["the"].values()), 4))
    print("  Test passes?", broken.check_normalisation(verbose=False))

    print()
    print("=" * 60)
    print("PART VIII / Q9: next-word prediction")
    print("=" * 60)
    for word in [START, "the", "cat", "sat", "on", "ran", "to"]:
        model.show_distribution(word)
        print("      arg max ->", model.most_probable(word))

    print()
    print("=" * 60)
    print("PART IX: 20 sampled sentences (seed=%d)" % seed)
    print("=" * 60)
    sampled = []
    for _ in range(20):
        sampled.append(model.generate("sample"))
    for i, s in enumerate(sampled, 1):
        print("%2d. %s" % (i, s))
    with open(os.path.join(out_dir, "generated_first_order.txt"), "w") as f:
        f.write("\n".join(sampled) + "\n")

    print()
    print("=" * 60)
    print("PART X / Q10: greedy vs sampling (5 each)")
    print("=" * 60)
    print("Greedy (capped at %d words):" % MAX_LEN)
    for _ in range(5):
        print("  ", model.generate("greedy"))
    print("Sampling:")
    for _ in range(5):
        print("  ", model.generate("sample"))

    print()
    print("Distinct parameters (non-zero CPT entries):", model.num_parameters())
    print("Saved: generated_first_order.txt")


if __name__ == "__main__":
    main()
