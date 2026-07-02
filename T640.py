import pandas as pd
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
import re

@dataclass
class T640:
    """
    Data class for T640 data.
    """

    file: Union[str, Path] = field(default=None,
        metadata={"description": ".xlsx file of T640 data."},
        )

    diameters: np.ndarray = field(
        default=None,
        metadata={"units": "µm", "description": "Dust particle diameters"},
    )

    bin_numbers: np.ndarray = field(
        default=None,
        metadata={"units": "-", "description": "Bin indices"},
    )

    bin_centers: np.ndarray = field(
        default=None,
        metadata={"units": "log(D[µm])", "description": "Centers of diameter bins"},
    )

    bin_edges: np.ndarray = field(
        default = None,
        metadata={"units": "log(D[µm])", "description": "Centers of diameter bins"},
    )

    timestamps: np.ndarray[Any, np.dtype[np.datetime64]] = field(
        default = None,
        metadata={"units": "-", "description": "Timestamps of measurements"},
    )

    counts: np.ndarray = field(
        default=None,
        metadata={"units": "#", "description": "Counts for each bin."}
    )

    num_concentration: np.ndarray = field(
        default=None,
        metadata={"units": "#/cm3", "description": "Count density in each bin"}
    )



    def import_data(self,file,return_DataFrame=False):
        df = pd.read_excel(file)
        pattern = re.compile(r'BIN(\d+)\((\d+\.\d+)\)')

        bins = []
        for col in df.columns:
            m = pattern.search(col)
            if m:
                bins.append((int(m.group(1)), float(m.group(2))))

        bin_nums  = [b for b, _ in bins]
        bin_sizes = [s for _, s in bins]
        
        new_col_names = ['Date_Time','NUMCONC']
        new_col_names.extend([f"BIN{b}" for b in bin_nums])
        df.columns = new_col_names

        self.diameters = np.array(bin_sizes)
        self.bin_centers = np.log10(bin_sizes)

        first_edge = self.bin_centers[0] - 0.5*(self.bin_centers[1]-self.bin_centers[0])
        other_edges = 0.5*(self.bin_centers[0:-1]+self.bin_centers[1::])
        last_edge = self.bin_centers[-1] + 0.5*(self.bin_centers[-1]-self.bin_centers[-2])
        self.bin_edges = np.r_[first_edge, other_edges, last_edge]

        self.timestamps = pd.to_datetime(df.Date_Time).to_numpy()
        self.bin_numbers = bin_nums
        self.file = file
        self.counts = df[[f"BIN{b}" for b in bin_nums]].to_numpy()
        self.num_concentration = df['NUMCONC'].to_numpy()

        if return_DataFrame:
            return df

    def sum_counts(self, start=None, end=None, inclusive="both"):
        """
        Sum bin counts over a time window.

        Parameters
        ----------
        start, end : str, datetime-like, or np.datetime64, optional
            Bounds of the window. Anything pd.to_datetime accepts is fine.
            If None, the window is open on that side (all data up to `end`,
            or all data from `start`).
        inclusive : {"both", "left", "right", "neither"}, default "both"
            Which endpoints to include, matching pandas conventions.

        Returns
        -------
        np.ndarray
            Per-bin summed counts, shape (n_bins,), aligned with
            self.bin_numbers / self.bin_centers. Bins with no samples in
            the window sum to 0.
        """
        if self.counts is None or self.timestamps is None:
            raise ValueError("No data loaded; call import_data() first.")

        ts = pd.to_datetime(self.timestamps)
        mask = np.ones(len(ts), dtype=bool)

        if start is not None:
            start = pd.to_datetime(start)
            mask &= (ts >= start) if inclusive in ("both", "left") else (ts > start)
        if end is not None:
            end = pd.to_datetime(end)
            mask &= (ts <= end) if inclusive in ("both", "right") else (ts < end)

        return self.counts[mask].sum(axis=0)

    def downsample(self, freq, label="left", closed="left",
                   include_num_concentration=True):
        """
        Downsample over fixed time windows: bin counts are summed, number
        concentration is averaged.

        Parameters
        ----------
        freq : str
            Resampling interval, any pandas offset alias, e.g. "5min",
            "1h", "15min", "1D".
        label, closed : {"left", "right"}, default "left"
            Which edge of each interval labels it and which edge is
            inclusive, matching pandas .resample() semantics.
        include_num_concentration : bool, default True
            If True and number concentration is available, append a
            "NUMCONC" column holding the window mean (a density, so it is
            averaged rather than summed).

        Returns
        -------
        T640
            A new T640 whose `timestamps` are the (timezone-naive) window
            labels and whose `counts` are the per-window bin sums. Bin
            metadata (`diameters`, `bin_numbers`, `bin_centers`,
            `bin_edges`, `file`) is carried over unchanged. If
            `include_num_concentration`, `num_concentration` holds the
            per-window mean; otherwise it is None. The result is shorter
            than the input by roughly the ratio of `freq` to the native
            sampling interval. Windows that genuinely recorded zero counts
            are 0; windows containing no samples at all (gaps in the
            record) are NaN.
        """
        if self.counts is None or self.timestamps is None:
            raise ValueError("No data loaded; call import_data() first.")

        # Rows with a missing timestamp can't be placed in a window; drop
        # them so they don't silently land in the wrong bucket.
        index = pd.to_datetime(self.timestamps)
        valid = index.notna()
        index = index[valid]
        bin_cols = [f"BIN{b}" for b in self.bin_numbers]
        df = pd.DataFrame(self.counts[valid], index=index, columns=bin_cols)

        resampler = df.resample(freq, label=label, closed=closed)
        out = resampler.sum()

        # Distinguish a true zero-count window from an empty (gap) window:
        # sum() reports both as 0, so NaN the windows that held no samples.
        empty = resampler.size() == 0
        out[empty] = np.nan

        num_conc = None
        if include_num_concentration and self.num_concentration is not None:
            nc = pd.Series(self.num_concentration[valid], index=index)
            num_conc = nc.resample(freq, label=label, closed=closed).mean().to_numpy()

        return T640(
            file=self.file,
            diameters=self.diameters,
            bin_numbers=self.bin_numbers,
            bin_centers=self.bin_centers,
            bin_edges=self.bin_edges,
            timestamps=out.index.to_numpy(),
            counts=out.to_numpy(),
            num_concentration=num_conc,
        )




    
        
