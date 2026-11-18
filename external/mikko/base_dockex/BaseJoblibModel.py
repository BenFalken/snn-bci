import joblib
import warnings

from base_dockex.BaseModel import BaseModel


class BaseJoblibModel(BaseModel):
    def load(self):
        print('Loading model')
        self.model = joblib.load(self.input_pathnames['model_joblib'])

    def save(self):
        if self.method == 'predict':
            warnings.warn(
                "User requested save_model when model was already loaded "
                "from file. Skipping model save."
            )
        else:
            print('Saving model')
            with open(self.output_pathnames['model_joblib'], 'wb') as model_file:
                joblib.dump(self.model, model_file)