import numpy as np
import random

from base_dockex.BaseDockex import BaseDockex


class BaseModel(BaseDockex):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.method = self.params['method']
        self.save_model = self.params['save_model']
        self.random_seed = self.params['random_seed']

        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None

        self.model = None

        self.predict_train = None
        self.predict_valid = None
        self.predict_test = None

    @staticmethod
    def _load_and_expand(pathname):
        array = np.load(pathname)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return array

    def set_random_seeds(self):
        print('Setting random seeds')
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def load_data(self):
        print('Loading data')

        if self.input_pathnames['X_train_npy'] is not None:
            self.X_train = np.load(self.input_pathnames['X_train_npy'])
        if self.input_pathnames['y_train_npy'] is not None:
            self.y_train = self._load_and_expand(self.input_pathnames['y_train_npy'])

        if self.input_pathnames['X_valid_npy'] is not None:
            self.X_valid = np.load(self.input_pathnames['X_valid_npy'])
        if self.input_pathnames['y_valid_npy'] is not None:
            self.y_valid = self._load_and_expand(self.input_pathnames['y_valid_npy'])

        if self.input_pathnames['X_test_npy'] is not None:
            self.X_test = np.load(self.input_pathnames['X_test_npy'])
        if self.input_pathnames['y_test_npy'] is not None:
            self.y_test = self._load_and_expand(self.input_pathnames['y_test_npy'])

    def instantiate_model(self):
        raise NotImplementedError("Subclasses of BaseModel must implement instantiate_model.")

    def load(self):
        raise NotImplementedError(
            "load() is not implemented on BaseModel. Use a subclass such as "
            "BaseJoblibModel, or provide your own load() override, to support "
            "method='predict'."
        )

    def save(self):
        raise NotImplementedError(
            "save() is not implemented on BaseModel. Use a subclass such as "
            "BaseJoblibModel, or provide your own save() override, to support "
            "save_model=true."
        )

    def fit(self):
        print('Fitting model')
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        print('Generating predictions')
        if self.X_train is not None:
            self.predict_train = self.model.predict(self.X_train)
        if self.X_valid is not None:
            self.predict_valid = self.model.predict(self.X_valid)
        if self.X_test is not None:
            self.predict_test = self.model.predict(self.X_test)

    def fit_predict(self):
        self.fit()
        self.predict()

    def write_outputs(self):
        print('Writing output arrays')
        if self.predict_train is not None:
            np.save(self.output_pathnames['predict_train_npy'], self.predict_train)
        if self.predict_valid is not None:
            np.save(self.output_pathnames['predict_valid_npy'], self.predict_valid)
        if self.predict_test is not None:
            np.save(self.output_pathnames['predict_test_npy'], self.predict_test)

    def run(self):
        print('Running')

        self.set_random_seeds()
        self.load_data()

        if self.method == 'fit':
            self.instantiate_model()
            self.fit()

        elif self.method == 'fit_predict':
            self.instantiate_model()
            self.fit_predict()

        elif self.method == 'predict':
            self.load()
            self.predict()

        else:
            raise ValueError(f"Received unsupported method: {self.method}")

        self.write_outputs()

        if self.save_model:
            self.save()

        print('Success')