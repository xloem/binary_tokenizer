import torch, transformers
from .tokenizer import BinaryTokenizer

# Goal: train an adapter, a model, and embeddings, together to produce the same output as the model.

def pipeline(task, model_path, adapter_path):
    pipeline = transformers.pipeline(task=task, model = model_path)
    try:
        adapter_name = pipeline.model.load_adapter(adapter_path)
    except:
        adapter_name = 'binary'
    return transformers.pipeline(task=task, model = BinaryAdaptedModel(pipeline.model, adapter_name), tokenizer = BinaryTokenizer(), prefix = b'')

class BinaryAdaptedModel(torch.nn.Module):
    def __init__(self, model, adapter_name, adapter_config = None):
        super().__init__()
        self.model = model
        self.config = model.config
        self.adapter_name = adapter_name
        try:
            self.model.set_active_adapters([self.adapter_name])
        except:
            print("creating new adapter " + adapter_name)
            if adapter_config is None:
                adapter_config = 'pfeiffer+inv'
            self.model.add_adapter(adapter_name, config = adapter_config)
            self.model.set_active_adapters([self.adapter_name])

    #def __init__(self, model_path, adapter_path, adapter_config):
        #self.model = transformers.AutoAdapterModel.from_pretrained(path + '/model', *params, **params)
        #self.adapter_name = self.model.load_adapter(path + '/adapter', config = adapter_config)
        #self.model.set_active_adapter(self.adapter_name)
        # should be 256 embeddings
        # note: +inv will train an embedding adapter that is faster
        # the output is then inverted
        
    def train_to_match(self, tokenizer, num_input_tokens, batch_size, grad_accum, optim, device='cuda'):
        def tokens_to_bytes(data):
            items=[
                torch.frombuffer(prompt, dtype=torch.uint8).to(int)
                for prompt in tokenizer.batch_decode(data)
            ]
            min_length = min((len(item) for item in items))
            return torch.stack([
                item[-min_length:] for item in items
            ])
        device = torch.Device(device)
        self.model.to(device)
        token_data = torch.randint(0, tokenizer.vocab_size, (batch_size, num_input_tokens))
        bytes_data = tokens_to_bytes(train_token_data)
        # first disable adapter and pass token data to produce labels
        # then re-enable adapter
        # then train bytes until last byte loss stops reducing
        # then add more data and repeat, i suppose

    def train(self):
        return self.model.train_adapter()

    def eval(self):
        return self.model.eval()

    def save_pretrained(self, path):
        self.model.save_adapter(path, self.adapter_name)
        
    @classmethod
    def from_pretrained(cls, model_path, adapter_path):
        model = transformers.AutoAdapterModel.from_pretrained(model_path)
        try:
            adapter_name = model.load_adapter(adapter_path)
        except:
            adapter_name = 'binary'
        return cls(model, adapter_name)
        #adapter_config = transformers.AdapterConfig.log(path + '/adapter')
        #return cls(model_path = path + '/model', adapter_path = path + '/adapter', adapter_config = adapter_config)

    def generate(self, *params, **kwparams):
        return self.model.generate(*params, **kwparams)
