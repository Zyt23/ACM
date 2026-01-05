from models import  timer_xl, moirai, moment, ttm, WeaverConv, WeaverMLP


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "timer_xl": timer_xl,
            "moirai": moirai,
            "moment": moment,
            "ttm": ttm,
        }
        self.adapter_dict = {
            "WeaverConv": WeaverConv,
            "WeaverMLP": WeaverMLP,
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
