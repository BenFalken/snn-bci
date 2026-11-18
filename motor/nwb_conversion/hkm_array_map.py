"""
Which Utah array (M1 or PMd) each unit of an HKM session was recorded on, in the units table's order: the order of
the channels in every dataset built from it (convert_nwb_trials_to_raw_h5.py keeps one channel per unit, in order).

Writes {session: {array: [channel indices]}} to --out and prints the counts per array, so a one-array subject can be
cut from the full datasets (make_hkm_array_subset.py) with the same trials and splits.

Usage (on Oscar, ann-env):
    python hkm_array_map.py $DATA_ROOT/raw/hkm/jenkins/*.nwb --out $DATA_ROOT/dataset/hkm/jenkins/array_map.json
"""

import argparse
import json
import os

from pynwb import NWBHDF5IO


def unit_arrays(nwbfile):
    """One array label per unit: the electrode group (or its location) of the unit's electrode."""
    units = nwbfile.units.to_dataframe()
    elec = nwbfile.electrodes.to_dataframe() if nwbfile.electrodes is not None else None

    def label(group, location=None):
        name = getattr(group, "name", None) or (group if isinstance(group, str) else None)
        return name or location or "unknown"

    if "electrode_group" in units.columns:
        return [label(g) for g in units["electrode_group"]]
    if "electrodes" in units.columns and elec is not None:
        out = []
        for e in units["electrodes"]:
            row = e.iloc[0] if hasattr(e, "iloc") else elec.loc[int(e[0] if hasattr(e, "__len__") else e)]
            out.append(label(row.get("group_name", None) or row.get("group", None), row.get("location", None)))
        return out
    raise ValueError(f"no electrode information in the units table (columns: {list(units.columns)})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("nwb", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = {}
    for path in sorted(a.nwb):
        session = os.path.splitext(os.path.basename(path))[0]
        with NWBHDF5IO(path, "r", load_namespaces=True) as io:
            f = io.read()
            labels = unit_arrays(f)
            groups = {k: (v.description, v.location) for k, v in f.electrode_groups.items()}
        # groups are named by number ('1', '2'); name each array by the area its description gives
        area = {k: next((a for a in ("M1", "PMd") if a in d), k) for k, (d, _) in groups.items()}
        arrays = {}
        for i, lab in enumerate(labels):
            arrays.setdefault(area.get(str(lab), str(lab)), []).append(i)
        result[session] = arrays
        print(session, {k: len(v) for k, v in arrays.items()}, "groups:", groups)
        for k, v in arrays.items():
            print(f"   {k}: channels {v[0]}..{v[-1]}" + ("" if v == list(range(v[0], v[-1] + 1)) else " (not contiguous)"))
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
