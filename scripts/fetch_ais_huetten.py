"""Fetch raw AIS positions for a time window and area from Huetten (2025), Baltic aisstream.io.

Source: Huetten, M. (2025). AIS data of the Baltic Sea, 29 July to 27 October 2024, from
aisstream.io. figshare, CC BY 4.0. https://doi.org/10.6084/m9.figshare.29062715
(file https://ndownloader.figshare.com/files/54720737, about 1.1 GB parquet, one row group per
day). Only the row groups that overlap the window are read, by HTTP range requests.

Usage (from the repository root):

  python3 scripts/fetch_ais_huetten.py --start 2024-08-12 --days 15 \\
      --bounds 4754000 5083000 3482000 3863000 --crs EPSG:3035 --out ais_lithuania.parquet

Writes the rows inside the bounds (in --crs) with columns mmsi, timestamp (Unix s), lat, lon,
sog (knots), type (AIS ship-type code, where reported), x, y (in --crs).
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import urllib.error
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer

URL = "https://ndownloader.figshare.com/files/54720737"


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests (the parquet reader seeks)."""

    def __init__(self, url: str):
        self.base = url
        self._resolve()
        self.pos = 0
        self.fetched = 0

    def _resolve(self):
        """Follow figshare's redirect to a signed storage link (they expire after a while)."""
        req = urllib.request.Request(self.base, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.url = r.geturl()
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        for attempt in range(3):
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code != 403 or attempt == 2:
                    raise
                self._resolve()  # the signed link expired: get a fresh one
        self.pos += len(data)
        self.fetched += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--start", required=True, help="UTC date, e.g. 2024-08-12")
    ap.add_argument("--days", type=int, required=True)
    ap.add_argument(
        "--bounds", type=float, nargs=4, required=True, metavar=("X0", "X1", "Y0", "Y1")
    )
    ap.add_argument("--crs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = int(dt.datetime.fromisoformat(args.start).replace(tzinfo=dt.timezone.utc).timestamp())
    t1 = t0 + args.days * 86400
    f = HTTPRangeFile(URL)
    pf = pq.ParquetFile(f)
    md = pf.metadata
    names = [md.row_group(0).column(j).path_in_schema for j in range(md.num_columns)]
    ts = names.index("timestamp")
    groups = [
        i
        for i in range(md.num_row_groups)
        if md.row_group(i).column(ts).statistics.max >= t0
        and md.row_group(i).column(ts).statistics.min < t1
    ]
    print(f"{len(groups)} row groups overlap the window")
    to_crs = Transformer.from_crs("EPSG:4326", args.crs, always_xy=True)
    x0, x1, y0, y1 = args.bounds
    parts = []
    for i in groups:
        tab = pf.read_row_group(i, columns=["mmsi", "timestamp", "lat", "lon", "sog", "type"])
        df = tab.to_pandas()
        df = df[(df.timestamp >= t0) & (df.timestamp < t1)]
        x, y = to_crs.transform(df.lon.to_numpy(), df.lat.to_numpy())
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        df = df[inside].assign(x=np.asarray(x)[inside], y=np.asarray(y)[inside])
        parts.append(df)
        print(f"row group {i}: {inside.sum()} rows inside; fetched {f.fetched / 1e6:.0f} MB so far")
    import pandas as pd

    out = pd.concat(parts, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), args.out)
    print(f"wrote {len(out)} rows, {out.mmsi.nunique()} vessels, to {args.out}")


if __name__ == "__main__":
    main()
