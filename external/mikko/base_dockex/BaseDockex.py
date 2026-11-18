import json
# base_dockex/BaseDockex.py — reconstructed from observed usage in
# librosa_mel_transformer.py, since the original ChrisHeelan/base_dockex
# GitHub repo is no longer available.


class NoneDefaultDict(dict):
    """
    A dict that returns None for missing keys instead of raising KeyError.

    This matches observed usage in downstream Dockex modules, e.g.:
        self.input_pathnames['transformer_joblib']
    is accessed and compared against None even when the module's JSON
    config never defines that key.
    """

    def __missing__(self, key):
        return None


class BaseDockex:
    """
    Dockex module base class.

    Reads a JSON file describing a Dockex experiment job and extracts
    the params, input_pathnames, and output_pathnames dicts.

    Arguments:
        input_args (list): Input arguments to the module, similar to
            sys.argv. Second element (input_args[1]) must be a string
            pathname to a JSON file with 'params', 'input_pathnames',
            and 'output_pathnames' keys.

    Subclasses must implement the run() method.
    """

    def __init__(self, input_args):
        json_pathname = input_args[1]

        with open(json_pathname, 'r') as json_file:
            job_dict = json.load(json_file)

        self.params = NoneDefaultDict(job_dict.get('params', {}))
        self.input_pathnames = NoneDefaultDict(job_dict.get('input_pathnames', {}))
        self.output_pathnames = NoneDefaultDict(job_dict.get('output_pathnames', {}))
        self.config = NoneDefaultDict(job_dict)

    def run(self):
        raise NotImplementedError("Subclasses of BaseDockex must implement the run method.")