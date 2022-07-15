import codecs

from rex.model import RexBaseModel
import pickle


def dump_model(model: RexBaseModel, file_name: str, verbose=True):
    pickle.dump(model, open(file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print('The dump has been saved as file', file_name)


def load_model(file_name: str) -> any:
    return pickle.load(open(file_name, 'rb'))


def encode_model(model: RexBaseModel) -> str:
    return codecs.encode(pickle.dumps(model), "base64").decode()


def decode_model(model: str) -> any:
    return pickle.loads(codecs.decode(model.encode(), "base64"))
