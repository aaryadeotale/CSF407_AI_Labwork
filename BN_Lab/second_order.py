"""
Second-order autoregressive language model (Bayesian network with
X(t-2) -> X(t) <- X(t-1)).

    P(Xt | X(t-2), X(t-1))  estimated from counts of observed triples.

Sentences are padded with TWO <START> tokens so the first word is predicted
from context (<START>, <START>) and the second from (<START>, w1).

Also runs the Part XIII comparison against the first-order model, so keep
first_order.py in the same folder.

Run:  python second_order.py            (optionally:  python second_order.py 123  -> seed)
"""

import os
import random
import sys
from collections import Counter, defaultdict

from first_order import DATASET, END, MAX_LEN, START, FirstOrderModel, tokenise


class SecondOrderModel:
    """P(Xt | Xt-2, Xt-1) estimated from trigram counts."""

    def __init__(self):
        # counts[(a, b)][c] = C(a, b, c)   <-- triple counts live here
        self.counts = defaultdict(Counter)
        # probabilities[(a, b)][c] = P(c | a, b)   <-- the CPT
        self.probabilities = {}

    # -- learning ------------------------------------------------------------
    def train(self, tokenised_sentences):
        for tokens in tokenised_sentences:
            padded = [START] + tokens  # two <START> tokens in total
            for i in range(len(padded) - 2):
                context = (padded[i], padded[i + 1])
                nxt = padded[i + 2]
                self.counts[context][nxt] += 1
        self._build_probabilities()

    def _build_probabilities(self):
        """P(c | a, b) = C(a, b, c) / sum_k C(a, b, k)"""
        self.probabilities = {}
        for context, next_counts in self.counts.items():
            total = sum(next_counts.values())
            dist = {}
            for nxt, count in next_counts.items():
                dist[nxt] = count / total
            self.probabilities[context] = dist

    # -- inspection ----------------------------------------------------------
    def distribution(self, context):
        return self.probabilities.get(context)

    def show_distribution(self, context):
        dist = self.distribution(context)
        label = "%s, %s" % context
        if dist is None:
            print("  P(next | %s): no observed transitions" % label)
            return
        print("  P(next | %s):" % label)
        for nxt in sorted(dist, key=dist.get, reverse=True):
            print("      %-8s %.4f" % (nxt, dist[nxt]))

    def vocabulary(self):
        vocab = set()
        for next_counts in self.counts.values():
            vocab.update(next_counts.keys())
        return vocab

    def num_parameters(self):
        return sum(len(d) for d in self.probabilities.values())

    # -- prediction ----------------------------------------------------------
    def most_probable(self, context):
        dist = self.distribution(context)
        if dist is None:
            return None
        best_word = None
        best_prob = -1.0
        for word in sorted(dist):
            if dist[word] > best_prob:
                best_word = word
                best_prob = dist[word]
        return best_word

    def sample_next(self, context):
        dist = self.distribution(context)
        if dist is None:
            return None
        words = list(dist.keys())
        weights = list(dist.values())
        return random.choices(words, weights=weights, k=1)[0]

    # -- generation ----------------------------------------------------------
    def generate(self, mode="sample", max_len=MAX_LEN):
        words = []
        context = (START, START)
        while len(words) < max_len:
            if mode == "greedy":
                nxt = self.most_probable(context)
            else:
                nxt = self.sample_next(context)
            if nxt is None or nxt == END:
                break
            words.append(nxt)
            context = (context[1], nxt)  # slide the window
        return " ".join(words)

    # -- testing -------------------------------------------------------------
    def check_normalisation(self, tolerance=1e-9, verbose=True):
        all_ok = True
        for context in sorted(self.probabilities):
            total = sum(self.probabilities[context].values())
            ok = abs(total - 1.0) < tolerance
            all_ok = all_ok and ok
            if verbose:
                print("  (%s, %s)  sum = %.6f  %s" % (context[0], context[1], total,
                                                       "OK" if ok else "FAIL"))
        return all_ok


# ----------------------------------------------------------------------------
# Part XIII: comparison helpers
# ----------------------------------------------------------------------------
def training_sentences():
    return set(DATASET)


def diversity_report(name, sentences):
    unique = set(sentences)
    novel = unique - training_sentences()
    print("  %-14s %2d generated | %2d distinct | %2d not in training data"
          % (name, len(sentences), len(unique), len(novel)))
    return unique, novel


def generate_many(model, n, mode="sample"):
    sentences = []
    for _ in range(n):
        sentences.append(model.generate(mode))
    return sentences


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    random.seed(seed)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    data = [tokenise(s) for s in DATASET]

    first = FirstOrderModel()
    first.train(data)
    second = SecondOrderModel()
    second.train(data)

    print("=" * 60)
    print("Second-order CPTs  P(next | previous two tokens)")
    print("=" * 60)
    for context in [(START, START), (START, "the"), ("the", "cat"),
                    ("the", "dog"), ("cat", "sat"), ("dog", "ran"),
                    ("on", "the"), ("to", "the")]:
        second.show_distribution(context)

    print()
    print("=" * 60)
    print("Normalisation test (every context must sum to 1)")
    print("=" * 60)
    print("All contexts sum to 1:", second.check_normalisation())

    print()
    print("=" * 60)
    print("Most probable next token for several contexts")
    print("=" * 60)
    for context in [(START, "the"), ("the", "cat"), ("the", "dog"),
                    ("sat", "on"), ("on", "the"), ("ran", "to")]:
        print("  arg max P(. | %s, %s) = %s" % (context[0], context[1],
                                               second.most_probable(context)))

    print()
    print("=" * 60)
    print("20 sampled sentences (seed=%d)" % seed)
    print("=" * 60)
    sampled = generate_many(second, 20, "sample")
    for i, s in enumerate(sampled, 1):
        print("%2d. %s" % (i, s))
    with open(os.path.join(out_dir, "generated_second_order.txt"), "w") as f:
        f.write("\n".join(sampled) + "\n")

    print()
    print("Greedy (capped at %d words):" % MAX_LEN)
    for _ in range(5):
        print("  ", second.generate("greedy"))

    # ---- Part XIII ---------------------------------------------------------
    print()
    print("=" * 60)
    print("PART XIII: first-order vs second-order")
    print("=" * 60)

    # V = number of real words; +1 because <START> can be a context and <END> a next token
    words = set()
    for tokens in data:
        for t in tokens:
            if t not in (START, END):
                words.add(t)
    v = len(words)
    n_context_tokens = v + 1   # words + <START>
    n_next_tokens = v + 1      # words + <END>

    first_full = n_context_tokens * n_next_tokens
    second_full = n_context_tokens * n_context_tokens * n_next_tokens
    first_contexts_seen = len(first.probabilities)
    second_contexts_seen = len(second.probabilities)
    first_contexts_possible = n_context_tokens
    second_contexts_possible = n_context_tokens * n_context_tokens  # upper bound

    print("  Vocabulary size (words only): %d" % v)
    print()
    print("  %-38s %10s %10s" % ("", "1st-order", "2nd-order"))
    print("  %-38s %10d %10d" % ("Distinct non-zero parameters",
                                 first.num_parameters(), second.num_parameters()))
    print("  %-38s %10d %10d" % ("Contexts observed in training",
                                 first_contexts_seen, second_contexts_seen))
    print("  %-38s %10d %10d" % ("Possible contexts (upper bound)",
                                 first_contexts_possible, second_contexts_possible))
    print("  %-38s %10d %10d" % ("Contexts with NO data (zero-prob)",
                                 first_contexts_possible - first_contexts_seen,
                                 second_contexts_possible - second_contexts_seen))
    print("  %-38s %10d %10d" % ("Full CPT size (contexts x next tokens)",
                                 first_full, second_full))
    print("  %-38s %9.1f%% %9.1f%%" % ("Fraction of full CPT that is non-zero",
                                       100.0 * first.num_parameters() / first_full,
                                       100.0 * second.num_parameters() / second_full))

    print()
    print("  Diversity over 200 sampled sentences each:")
    first_sents = generate_many(first, 200, "sample")
    second_sents = generate_many(second, 200, "sample")
    first_unique, first_novel = diversity_report("first-order", first_sents)
    second_unique, second_novel = diversity_report("second-order", second_sents)

    print()
    print("  Novel sentences (not in training data) -- first-order:")
    for s in sorted(first_novel):
        print("     -", s)
    print("  Novel sentences (not in training data) -- second-order:")
    if second_novel:
        for s in sorted(second_novel):
            print("     -", s)
    else:
        print("     (none: the second-order model only reproduces training sentences here)")

    print()
    print("  Distinct sentences the second-order model can produce:",
          len(second_unique), "of the", len(training_sentences()), "training sentences")
    print("Saved: generated_second_order.txt")


if __name__ == "__main__":
    main()
