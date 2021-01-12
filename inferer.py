import torch

from utils.config import device


class ModelInferer:
    def __init__(self, config, checkpoint_path, quantize=True):
        self.device = device

        model_config = config.model_config.from_pretrained(
            config["model_name"],
            num_labels=config["num_labels"],
            finetuning_task=config["task_name"]
        )
        model_config.update(config)

        self.model_type = model_config.model_type
        self.tokenizer = config.tokenizer_class.from_pretrained(config["tokenizer_name"])
        self.model = config.model_class(model_config).to(device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        if quantize:
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)

        self.model.to(device)
        self.model.eval()

        # file to map output to category name
        self.mapping = {key: value for key, value in enumerate(sorted(config["labels"]))}

        torch.set_num_threads(1)

    def preprocess(self, text):
        """"""
        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=512
        )
        return tokens

    def inference(self, inputs):
        """"""
        if 'distilbert' not in self.model_type:
            token_type_ids = inputs['token_type_ids'].to(self.device) if self.model_type in ["bert", "xlnet"] else None
            output = self.model(
                inputs['input_ids'].to(self.device),
                token_type_ids=token_type_ids
            )
        else:
            output = self.model(
                inputs['input_ids'].to(self.device)
            )
        sigmoid = torch.sigmoid(output[0]).squeeze(0)
        prediction = (sigmoid > 0.5).nonzero().flatten().tolist()
        prediction = [self.mapping[p] for p in prediction]
        return prediction

    def predict(self, text):
        """"""
        inputs = self.preprocess(text)
        return self.inference(inputs)
