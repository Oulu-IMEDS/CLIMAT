from .climat import CLIMAT
from .fcn import FCN
from .mmtf import Multimodal_Transformer
from .recurrent import BiRecurrent_Model


def create_model(cfg, device, pn_weights=None, y0_weights=None):
    if cfg.method_name == "fcn":
        return FCN(cfg, device, pn_weights=pn_weights)
    elif cfg.method_name in ["gru", "lstm"]:
        return BiRecurrent_Model(cfg, device, pn_weights=pn_weights)
    elif cfg.method_name == "mmtf":
        return Multimodal_Transformer(cfg, device, pn_weights=pn_weights)
    elif cfg.method_name == "climat":
        return CLIMAT(cfg, device, pn_weights=pn_weights, y0_weights=y0_weights)
    else:
        raise ValueError(f"Not support method name '{cfg.method_name}'.")
