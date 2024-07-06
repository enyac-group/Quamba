import unittest
import sys
sys.path.append('.')
import torch
from quamba.fake_quant.smooth_quant_utils import SmoothModule, smooth_mamba
from quamba.fake_quant.qLinearLayer import QLinearLayer
from quamba.fake_quant.configuration_jamba import JambaConfig
from quamba.fake_quant.jamba_simple import JambaMambaMixer
from quamba.fake_quant.qJamba import QJambaMambaMixer
from transformers import AutoTokenizer

class SmoothQuantModuleTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.weight = torch.rand(10,10)
        self.x = torch.rand(10,1)
        self.mod = SmoothModule(self.weight, tensor_name="weight")

        return super().setUp()
    
    def test_before_activation(self):
        result = self.mod(self.x)
        self.assertIs(self.x, result)

    def test_configure(self):
        # test with scale vector
        scales = torch.rand(10,1)
        self.mod.configure(scales)

        self.assertTrue(self.mod.activated)
        result = self.mod(self.x)
        self.assertTrue((result == self.x.div(scales)).all())

        # test with scale integer
        scales = torch.rand(1,1)
        self.mod.configure(scales)

        self.assertTrue(self.mod.activated)
        result = self.mod(self.x)
        self.assertTrue((result == self.x.div(scales)).all())

class SmoothMambaTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config = JambaConfig()
        # self.model = JambaModel(config)
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.mamba = JambaMambaMixer(config, layer_idx=0)
        self.model = torch.nn.Sequential(self.embed_tokens, self.mamba)
        self.model = self.model.to("cuda")

        self.quant_mamba = QJambaMambaMixer(self.mamba)
        self.quant_model = torch.nn.Sequential(self.embed_tokens, self.quant_mamba)
        self.tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")
        return super().setUp()

    def test_smooth_base(self) -> None:
        # ensure the base model does not cause error when it does 
        # not have SmoothModule or QLinear
        smooth_mamba(self.model, self.tokenizer, num_samples=50)
    
    def test_smooth_quant(self) -> None:
        weights = []
        for name, m in self.quant_model.named_modules():
            if isinstance(m, SmoothModule):
                name_prefix = ".".join(name.split(".")[:-1])
                weight_name = name_prefix + "." + m.weight_to_smooth
                linear = self.quant_model.get_submodule(weight_name)
                weights.append(linear.weight.clone())
        smooth_mamba(self.quant_model, self.tokenizer, num_samples=50)
        i = 0
        for name, m in self.quant_model.named_modules():
            if isinstance(m, SmoothModule):
                print("evaluating SmoothModule")
                name_prefix = ".".join(name.split(".")[:-1])
                weight_name = name_prefix + "." + m.weight_to_smooth
                linear = self.quant_model.get_submodule(weight_name)
                self.assertTrue(torch.allclose(weights[i], linear.weight.div(m.scales)))
                i += 1
def suite():
    suite = unittest.TestSuite()
    suite.addTest(SmoothQuantModuleTestCase('test_before_activation'))
    suite.addTest(SmoothQuantModuleTestCase('test_configure'))
    suite.addTest(SmoothMambaTestCase('test_smooth_base'))
    suite.addTest(SmoothMambaTestCase('test_smooth_quant'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())