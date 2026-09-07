"""Python half of the format cross-check.

Loads a model written by Julia using ONLY tools/read_tmcore.py and the format spec, then
independently reproduces Julia's predictions. Agreement means the format is genuinely readable by a
second implementation rather than merely round-tripping through its own writer.

    python compare.py model.tmc cases.txt
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import read_tmcore  # noqa: E402


def main(model_path, cases_path):
    m = read_tmcore.load(model_path)
    print("python: loaded %d classes, %d clauses/polarity, width %d, LF %d, ceiling=%s"
          % (m.nclasses, m.nclauses, m.width, m.LF, read_tmcore.CEILING[m.ceiling]))

    # cases.txt: one line per example, "<bitstring> <julia_prediction> <julia_scores...>"
    lines = [ln.split() for ln in open(cases_path) if ln.strip()]
    mismatched_pred, mismatched_score, checked = 0, 0, 0

    for parts in lines:
        bits = [ch == "1" for ch in parts[0]]
        want_pred = int(parts[1])
        want_scores = [int(v) for v in parts[2:]]
        if len(bits) != m.width:
            raise SystemExit("case width %d does not match model width %d" % (len(bits), m.width))

        chunks = [0] * m.nchunks
        for i, b in enumerate(bits):
            if b:
                chunks[i >> 6] |= 1 << (i & 63)

        got_scores = []
        for ci in range(m.nclasses):
            pos = sum(m.clause_vote(ci, 0, j, chunks) for j in range(m.nclauses))
            neg = sum(m.clause_vote(ci, 1, j, chunks) for j in range(m.nclauses))
            got_scores.append(pos - neg)

        best = max(range(m.nclasses), key=lambda ci: (got_scores[ci], -ci))
        got_pred = m.classes[best]

        checked += 1
        if got_scores != want_scores:
            mismatched_score += 1
        if got_pred != want_pred:
            mismatched_pred += 1

    print("python: %d cases, %d score mismatches, %d prediction mismatches"
          % (checked, mismatched_score, mismatched_pred))
    return 0 if (mismatched_score == 0 and mismatched_pred == 0) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
