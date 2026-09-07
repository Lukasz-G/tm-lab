"""Reference reader for the TMCore model format, version 3.

Pure standard library, no NumPy, no Julia. If this file is hard to follow, the format is wrong —
its entire purpose is that another implementation can adopt it without adopting us.

The spec is docs/model-format.md. This is a second implementation of that document, not a port of
the writer, and it is checked against the Julia writer on a real trained model.

    python tools/read_tmcore.py model.tmc            # summary
    python tools/read_tmcore.py model.tmc --clause 0 3   # literals of one clause
"""

import struct
import sys

MAGIC = b"TMCORE\0\0"
VERSION = 3

CEILING = {0: "literal-capped", 1: "flat-LF"}
BUDGET = {0: "growth-gate", 1: "hard-cap"}
FEEDBACK = {0: "threshold", 1: "proportional", 2: "proportional-idle"}
MISSCOST = {0: "uniform", 1: "confidence-weighted"}


class Model:
    """A loaded model. Clause banks live in `self.banks[class_index][polarity]`."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def clause_literals(self, class_index, polarity, j):
        """Included literals of one clause as (positive, negated) lists of 1-based positions."""
        inc, invm = self.banks[class_index][polarity]["included"], \
                    self.banks[class_index][polarity]["included_inv"]
        pos, neg = [], []
        for n in range(self.nchunks):
            for b in range(64):
                i = n * 64 + b + 1
                if i > self.width:
                    break
                if inc[j][n] >> b & 1:
                    pos.append(i)
                if invm[j][n] >> b & 1:
                    neg.append(i)
        return pos, neg

    def clause_vote(self, class_index, polarity, j, chunks):
        """Fuzzy vote of one clause on a packed input, straight from the spec."""
        bank = self.banks[class_index][polarity]
        weighted = self.misscost == 1
        if weighted and self.state_bytes == 0:
            raise ValueError("confidence-weighted scoring needs automata, which this file omits")
        mid = self.include_limit + (self.state_max - self.include_limit) // 2
        misses = 0
        for n in range(self.nchunks):
            inc, invm = bank["included"][j][n], bank["included_inv"][j][n]
            val = (((inc ^ invm) & chunks[n]) ^ inc) & 0xFFFFFFFFFFFFFFFF
            if not weighted:
                misses += bin(val).count("1")
                continue
            while val:
                b = (val & -val).bit_length() - 1
                i = n * 64 + b
                confident = ((inc >> b & 1 and bank["state"][j][i] >= mid) or
                             (invm >> b & 1 and bank["state_inv"][j][i] >= mid))
                misses += 2 if confident else 1
                val &= val - 1
        count = bank["count"][j]
        if self.ceiling == 0:                      # literal-capped
            ceiling = self.LF if count == 0 else min(count, self.LF)
        else:                                      # flat LF
            ceiling = self.LF
        return max(0, ceiling - misses)

    def predict(self, bits):
        """Predict one example given a list/sequence of bools of length `width`."""
        chunks = [0] * self.nchunks
        for i, b in enumerate(bits):
            if b:
                chunks[i >> 6] |= 1 << (i & 63)
        best_i, best_v = 0, None
        for ci in range(self.nclasses):
            v = sum(self.clause_vote(ci, 0, j, chunks) for j in range(self.nclauses)) - \
                sum(self.clause_vote(ci, 1, j, chunks) for j in range(self.nclauses))
            if best_v is None or v > best_v:
                best_i, best_v = ci, v
        return self.classes[best_i]


def _u32(buf, off):
    return struct.unpack_from("<I", buf, off)[0]


def load(path):
    with open(path, "rb") as fh:
        buf = fh.read()

    if buf[:8] != MAGIC:
        raise ValueError("not a TMCore model file")
    version = _u32(buf, 8)
    if version != VERSION:
        raise ValueError("unsupported format version %d (this reader handles %d)" % (version, VERSION))

    header_size = _u32(buf, 12)
    width, nchunks, nclasses, nclauses = (_u32(buf, o) for o in (16, 20, 24, 28))
    T, S, L, LF = (_u32(buf, o) for o in (32, 36, 40, 44))
    include_limit, state_min, state_max = (_u32(buf, o) for o in (48, 52, 56))
    ceiling, budget, state_bytes, class_type = struct.unpack_from("<BBBB", buf, 60)
    payload_bytes = struct.unpack_from("<Q", buf, 64)[0]
    class_block_bytes = _u32(buf, 72)
    feedback = buf[76]
    misscost = buf[77]
    if any(buf[78:80]):
        raise ValueError("reserved header bytes are not zero")

    off = 80
    classes = []
    if class_type == 1:
        for _ in range(nclasses):
            n = _u32(buf, off); off += 4
            classes.append(buf[off:off + n].decode("utf-8")); off += n
    else:
        for _ in range(nclasses):
            v = struct.unpack_from("<q", buf, off)[0]; off += 8
            classes.append(bool(v) if class_type == 2 else v)

    if off != header_size:
        raise ValueError("class block ended at %d, header says payload starts at %d" % (off, header_size))
    if len(buf) - header_size != payload_bytes:
        raise ValueError("payload is %d bytes, header declares %d"
                         % (len(buf) - header_size, payload_bytes))

    state_fmt = {0: None, 1: "<%dB", 2: "<%dH"}[state_bytes]

    def read_bank():
        nonlocal off
        bank = {}
        for key in ("included", "included_inv"):
            rows = []
            for _ in range(nclauses):
                rows.append(list(struct.unpack_from("<%dQ" % nchunks, buf, off)))
                off += 8 * nchunks
            bank[key] = rows
        bank["count"] = list(struct.unpack_from("<%di" % nclauses, buf, off))
        off += 4 * nclauses
        if state_fmt:
            for key in ("state", "state_inv"):
                rows = []
                for _ in range(nclauses):
                    rows.append(list(struct.unpack_from(state_fmt % width, buf, off)))
                    off += state_bytes * width
                bank[key] = rows
        # The stored count must agree with the masks; disagreeing means a broken writer.
        for j in range(nclauses):
            popcount = sum(bin(c).count("1") for c in bank["included"][j]) + \
                       sum(bin(c).count("1") for c in bank["included_inv"][j])
            if popcount != bank["count"][j]:
                raise ValueError("clause %d: stored count %d but masks hold %d literals"
                                 % (j, bank["count"][j], popcount))
        return bank

    banks = [[read_bank(), read_bank()] for _ in range(nclasses)]
    if off != len(buf):
        raise ValueError("%d trailing bytes after payload" % (len(buf) - off))

    return Model(width=width, nchunks=nchunks, nclasses=nclasses, nclauses=nclauses,
                 T=T, S=S, L=L, LF=LF, include_limit=include_limit,
                 state_min=state_min, state_max=state_max,
                 ceiling=ceiling, budget=budget, feedback=feedback, misscost=misscost,
                 state_bytes=state_bytes,
                 classes=classes, banks=banks)


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    m = load(argv[0])
    print("TMCore model, format v%d" % VERSION)
    print("  width %d bits (%d chunks), %d classes, %d clauses per polarity"
          % (m.width, m.nchunks, m.nclasses, m.nclauses))
    print("  T %d  S %d  L %d  LF %d" % (m.T, m.S, m.L, m.LF))
    print("  ceiling %s, budget %s, feedback %s, miss-cost %s, automata %s"
          % (CEILING.get(m.ceiling, "?"), BUDGET.get(m.budget, "?"),
             FEEDBACK.get(m.feedback, "?"), MISSCOST.get(m.misscost, "?"),
             "absent" if m.state_bytes == 0 else "%d-bit" % (m.state_bytes * 8)))
    print("  classes: %s" % (m.classes,))
    counts = [c for cls in m.banks for pol in cls for c in pol["count"]]
    counts.sort()
    print("  literals per clause: min %d, median %d, max %d"
          % (counts[0], counts[len(counts) // 2], counts[-1]))
    if "--clause" in argv:
        i = argv.index("--clause")
        ci, j = int(argv[i + 1]), int(argv[i + 2])
        pos, neg = m.clause_literals(ci, 0, j)
        print("\nclass %s, positive clause %d:" % (m.classes[ci], j))
        print("  requires set:   %s" % pos)
        print("  requires clear: %s" % neg)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
