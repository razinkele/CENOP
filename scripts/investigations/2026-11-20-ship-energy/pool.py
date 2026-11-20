import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/investigations/2026-11-20-ship-energy")
import common as c

S = Path(sys.argv[1])
j = dict(np.load(S / "java.npz"))
j["strength"] = j["strength"] / 400.0  # the report proxy writes strength in metres
c.table("DEPONS (200 tracked animals x 8 seeds; strength / 400)", j)
acc = c.empty()
for f in sorted(S.glob("cenop-*.npz")):
    d = np.load(f)
    for k in c.KEYS:
        acc[k] += d[k]
c.table("CENOP (all animals, 4 seeds)", acc)
