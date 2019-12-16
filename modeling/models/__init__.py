from .setnet import SetNet

_dict = dict(
    SetNet=SetNet,
)

def get_model(name):
    return _dict[name]