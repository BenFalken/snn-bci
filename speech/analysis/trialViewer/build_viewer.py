#builds the interactive trial viewer: viewer_template.html with the exported trials inlined, one self-contained page
#
#   python build_viewer.py trial_*.json --out ../../figures/viewer/trial_viewer.html
#exported trials come from plotExampleTrial.py --exportJson <folder> (on Oscar, see plotTrials.sbatch EXPORT_DIR)
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('trials', nargs='+', help='exported trial JSON files, shown in this order')
    p.add_argument('--out', default='trial_viewer.html')
    a = p.parse_args()
    trials = [json.load(open(f)) for f in a.trials]
    template = open(os.path.join(HERE, 'viewer_template.html')).read()
    data = json.dumps(trials, separators=(',', ':')).replace('</', '<\\/')
    html = template.replace('/*TRIALS*/[]', data)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, 'w').write(html)
    print(f'wrote {a.out} ({len(trials)} trials, {len(html) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
