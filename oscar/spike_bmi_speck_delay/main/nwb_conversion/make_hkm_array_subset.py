"""
A one-array HKM subject cut from the full one: the same sessions, trials and train/test splits, keeping only the
channels of one Utah array (from hkm_array_map.py), so every decoder runs on it unchanged as its own subject.

Jenkins has two 96-channel arrays (M1 and PMd). jenkins_m1 is the M1 array alone: 96 channels, like indy and loco,
so the Speck SNN is the bmi one exactly (speck_delay: 96 x 4 delays = 384 chip inputs -> 128 -> 256 -> 256 -> 36).

Writes, for --dst (e.g. jenkins_m1):
    dataset/hkm/<dst>/mua/<session>_binning.h5     every dataset with a channel axis cut to the array's channels
                                                   (other datasets and all attributes copied as they are)
    snn_datasets/hkm/<dst>/mua/<session>/{train,test}/*.pkl   input_spikes[channels]
    snn_training/velocity_scalers.json             hkm/<dst> = hkm/<src> (the velocities are the same)

Usage (on Oscar):
    python make_hkm_array_subset.py --src jenkins --dst jenkins_m1 --array M1 \
        --map $DATA_ROOT/dataset/hkm/jenkins/array_map.json
"""

import argparse
import glob
import json
import os
import pickle

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def pick(arrays, want):
    """The channel list of the array named like `want` (case-insensitive substring, e.g. M1 or PMd)."""
    hits = [k for k in arrays if want.lower() in k.lower()]
    if len(hits) != 1:
        raise ValueError(f"array {want!r} matches {hits} of {list(arrays)}")
    return np.asarray(arrays[hits[0]])


def cut_h5(src, dst, ch, n_all):
    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        fo.attrs.update(fi.attrs)

        def copy(name, obj):
            if isinstance(obj, h5py.Group):
                fo.require_group(name).attrs.update(obj.attrs)
                return
            data = obj[()]
            axes = [i for i, s in enumerate(np.shape(data)) if s == n_all]
            if axes:
                data = np.take(data, ch, axis=axes[-1])
            fo.create_dataset(name, data=data, compression="gzip" if np.ndim(data) > 1 else None)
            fo[name].attrs.update(obj.attrs)
            print(f"   {name}: {obj.shape} -> {np.shape(data)}")

        fi.visititems(copy)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="jenkins")
    p.add_argument("--dst", default="jenkins_m1")
    p.add_argument("--array", default="M1")
    p.add_argument("--map", required=True)
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", os.path.expandvars("/users/$USER/scratch/$USER/data")))
    a = p.parse_args()
    amap = json.load(open(a.map))
    root = a.data_root
    for h5 in sorted(glob.glob(f"{root}/dataset/hkm/{a.src}/mua/*_binning.h5")):
        session = os.path.basename(h5)[: -len("_binning.h5")]
        arrays = amap[session]
        n_all = sum(len(v) for v in arrays.values())
        ch = pick(arrays, a.array)
        print(f"{session}: {a.array} = {len(ch)} of {n_all} channels")
        os.makedirs(f"{root}/dataset/hkm/{a.dst}/mua", exist_ok=True)
        cut_h5(h5, f"{root}/dataset/hkm/{a.dst}/mua/{os.path.basename(h5)}", ch, n_all)
        for split in ("train", "test"):
            sdir = f"{root}/snn_datasets/hkm/{a.src}/mua/{session}/{split}"
            ddir = f"{root}/snn_datasets/hkm/{a.dst}/mua/{session}/{split}"
            os.makedirs(ddir, exist_ok=True)
            files = sorted(glob.glob(f"{sdir}/*.pkl"))
            for f in files:
                d = pickle.load(open(f, "rb"))
                assert d["input_spikes"].shape[0] == n_all, (f, d["input_spikes"].shape)
                d["input_spikes"] = d["input_spikes"][ch]
                pickle.dump(d, open(f"{ddir}/{os.path.basename(f)}", "wb"))
            print(f"   snn {split}: {len(files)} trials")
    vs = os.path.join(HERE, "..", "snn_training", "velocity_scalers.json")
    s = json.load(open(vs))
    s["hkm"][a.dst] = s["hkm"][a.src]
    json.dump(s, open(vs, "w"), indent=2)
    print(f"velocity_scalers.json: hkm/{a.dst} = hkm/{a.src}")


if __name__ == "__main__":
    main()
