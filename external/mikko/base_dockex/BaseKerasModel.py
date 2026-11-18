from tensorflow.keras.models import load_model

from base_dockex.BaseModel import BaseModel


class BaseKerasModel(BaseModel):
    """
    Dockex BaseModel subclass for Keras models — reconstructed from README
    documentation and observed usage in modules/decoders/keras/dense_nn/dense_nn.py.

    Subclasses must implement instantiate_model(), which must set self.model
    to a compiled Keras model, and may append to self.callbacks (already
    initialized as an empty list) to add training callbacks such as
    EarlyStopping.
    """

    def __init__(self, input_args):
        super().__init__(input_args)

        self.epochs = self.params['epochs']
        self.batch_size = self.params['batch_size']

        self.callbacks = []

    def fit(self):
        if self.input_pathnames['init_model_keras'] is not None:
            # fine-tuning: start from a saved model's weights (same architecture), e.g. one pretrained on a
            # larger dataset
            self.model.set_weights(load_model(self.input_pathnames['init_model_keras']).get_weights())
            print(f"Starting from the weights in {self.input_pathnames['init_model_keras']}")
        print('Fitting model')

        validation_data = None
        if self.X_valid is not None and self.y_valid is not None:
            validation_data = (self.X_valid, self.y_valid)

        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=self.callbacks,
            verbose=2,
        )

    def load(self):
        print('Loading model')
        self.model = load_model(self.input_pathnames['model_keras'])

    def save(self):
        if self.method == 'predict':
            import warnings
            warnings.warn(
                "User requested save_model when model was already loaded "
                "from file. Skipping model save."
            )
        else:
            print('Saving model')
            self.model.save(self.output_pathnames['model_keras'])