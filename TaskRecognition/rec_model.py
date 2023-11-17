import argparse
import os
import sys

from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from TaskRecognition.model.model import T5FineTuner


class ExtractNER:

    def __init__(self):
        model_path = "models/feature-ext.ckpt"

        args_dict = dict(
            model_name_or_path='t5-small',
            tokenizer_name_or_path='t5-small',
            check_point=model_path,
            max_seq_length=80,
        )
        args = argparse.Namespace(**args_dict)
        self.model = T5FineTuner(args).load_from_checkpoint(args.check_point)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    def predict_ner(self, input_sample):
        self.model.model.eval()
        model = self.model.to("cpu")
        tokenized_inputs = self.tokenizer.encode(input_sample.lower(), return_tensors="pt")

        model_output = model.model.generate(input_ids=tokenized_inputs)
        predictions = [self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                       .strip() for ids in model_output]
        texts = self.tokenizer.decode(tokenized_inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        predictions = [pred.strip() for pred in predictions[0].split(';') if ':' in pred]
        predictions = [pred.strip() for pred in predictions if pred.split(':')[0].strip() and pred.split(':')[1].strip()]
        predictions = set(predictions)
        # print(predictions)

        feature = []
        if predictions:
            for prediction in predictions:
                prediction = prediction.split(':')
                feature.append({prediction[0].strip(): prediction[1].strip()})

        return {"input": texts, "ner": feature}


if __name__ == "__main__":
    ner = ExtractNER()
    res = ner.predict_ner("We want to look at this perovskite to understand its structure. We think 5 seconds of exposure should be sufficient. Theta of 0.2 would be good.")
    # print(res)