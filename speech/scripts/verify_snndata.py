"""
Check a rebuilt derived/snnData against fingerprints of the copy that trained rounds 1-17 (taken on the Mac from
~/Downloads/speech_speck/snnData: test.npz of all 24 sessions, train.npz of 6). Every fingerprinted file must exist
and match (trials, bins and label sum exactly; feature sums to 1e-4 relative); every one of the 19 training sessions
must also have a train.npz. Exit code 1 on any failure.

    python verify_snndata.py /users/$USER/scratch/speechBCI_dryad/derived/snnData
"""
import json
import os
import sys

import numpy as np

S19 = ['t12.2022.04.28', 't12.2022.05.05', 't12.2022.05.17', 't12.2022.05.19', 't12.2022.05.24', 't12.2022.05.26',
       't12.2022.06.02', 't12.2022.06.07', 't12.2022.06.14', 't12.2022.06.16', 't12.2022.06.21', 't12.2022.06.28',
       't12.2022.07.05', 't12.2022.07.14', 't12.2022.07.21', 't12.2022.07.27', 't12.2022.08.02', 't12.2022.08.11',
       't12.2022.08.13']


def main(root):
    exp = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snndata_expected.json')))
    bad = 0
    for rel, e in exp.items():
        f = os.path.join(root, rel)
        if not os.path.isfile(f):
            print(f'MISSING  {rel}'); bad += 1; continue
        z = np.load(f)
        got = dict(trials=len(z['featureLens']), bins=int(z['featureLens'].sum()),
                   feat_sum=float(z['features'].astype(np.float64).sum()),
                   feat_abs=float(np.abs(z['features'].astype(np.float64)).sum()),
                   label_sum=int(z['labels'].astype(np.int64).sum()))
        ok = all(got[k] == e[k] for k in ('trials', 'bins', 'label_sum')) and \
            all(abs(got[k] - e[k]) <= 1e-4 * max(1.0, abs(e['feat_abs'])) for k in ('feat_sum', 'feat_abs'))
        print(f"{'ok      ' if ok else 'DIFFERS '} {rel}  trials {got['trials']}  bins {got['bins']}")
        bad += not ok
    for s in S19:
        if not os.path.isfile(os.path.join(root, s, 'train.npz')):
            print(f'MISSING  {s}/train.npz (training session)'); bad += 1
    print(f'{len(exp)} fingerprinted files checked, 19 training sessions checked: '
          + ('ALL MATCH' if not bad else f'{bad} PROBLEM(S)'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
