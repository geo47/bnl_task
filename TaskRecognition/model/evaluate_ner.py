import argparse

from datasets import load_metric, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from TaskRecognition.model.entity_dataset import EntityDataset
from TaskRecognition.model.model import T5FineTuner
from TaskRecognition.model.util import set_seed


class EvaluationNER:

    def __init__(self, arguments):
        self.args = argparse.Namespace(**arguments)
        self.model = T5FineTuner(self.args).load_from_checkpoint(self.args.check_point)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path)

    def save_model(self):
        self.model.model.save_pretrained("trained_T5")

    def find_sub_list(self, sl, l):
        results = []
        sll = len(sl)
        for ind in (i for i, e in enumerate(l) if e == sl[0]):
            if l[ind:ind + sll] == sl:
                results.append((ind, ind + sll - 1))
        return results

    def generate_label(self, input: str, target: str):
        mapper = {'O': 0, 'B-TASK_NO': 1, 'I-TASK_NO': 2, 'B-SAMPLE_NAME': 3, 'I-SAMPLE_NAME': 4, 'B-EXPOSURE_TIME': 5,
                  'I-EXPOSURE_TIME': 6, 'B-ANGLE': 7, 'I-ANGLE': 8}
        inv_mapper = {v: k for k, v in mapper.items()}

        input = input.split(" ")
        target = target.split("; ")

        init_target_label = [mapper['O']] * len(input)

        for ent in target:
            ent = ent.split(": ")
            # print(ent)
            # print(input)
            try:
                sent_end = ent[1].split(" ")
                index = self.find_sub_list(sent_end, input)
            except:
                continue
            # print(index)
            try:
                init_target_label[index[0][0]] = mapper[f"B-{ent[0].upper()}"]
                for i in range(index[0][0] + 1, index[0][1] + 1):
                    init_target_label[i] = mapper[f"I-{ent[0].upper()}"]
            except:
                continue
        init_target_label = [inv_mapper[j] for j in init_target_label]
        # print(init_target_label)
        return init_target_label

    def evaluate_label(self, test_dataset):

        dataset_test = EntityDataset(
            tokenizer=self.tokenizer, dataset=test_dataset, type_path='train', max_len=self.args.max_seq_length)
        print(len(dataset_test))
        d_train = dataset_test[0]
        print(self.tokenizer.decode(d_train["source_ids"], skip_special_tokens=False))
        print(self.tokenizer.decode(d_train["target_ids"], skip_special_tokens=False))

        dataloader = DataLoader(dataset_test, batch_size=self.args.eval_batch_size, num_workers=2)

        self.model.model.eval()
        model = self.model.to("cuda")
        true_labels = []
        pred_labels = []
        for batch in tqdm(dataloader):
            input_ids = batch['source_ids'].to("cuda")
            attention_mask = batch['source_mask'].to("cuda")
            outs = model.model.generate(input_ids=input_ids, attention_mask=attention_mask)
            predictions = [self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                           .strip() for ids in outs]
            targets = [self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                      for ids in batch["target_ids"]]
            texts = [self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                     for ids in batch["source_ids"]]
            true_label = [self.generate_label(texts[i].strip(), targets[i].strip())
                          if targets[i].strip() != 'none' else ["O"] * len(texts[i].strip().split())
                          for i in range(len(texts))]
            pred_label = [self.generate_label(texts[i].strip(), predictions[i].strip())
                          if predictions[i].strip() != 'none' else ["O"] * len(texts[i].strip().split())
                          for i in range(len(texts))]

            true_labels.extend(true_label)
            pred_labels.extend(pred_label)
            metric = load_metric("seqeval")

            for i in range(4):
                if i < len(texts):
                    print(f"Text:  {texts[i]}")
                    print(f"Predictions:  {predictions[i]}")
                    print(f"Targets:  {targets[i]}")
                    print(f"Predicted Token Class:  {pred_label[i]}")
                    print(f"True Token Class:  {true_label[i]}")
                    print("=====================================================================\n")

            print(metric.compute(predictions=pred_labels, references=true_labels))

    def evaluate(self, test_dataset):

        dataset_test = EntityDataset(
            tokenizer=self.tokenizer, dataset=test_dataset, type_path='train', max_len=self.args.max_seq_length)

        print(len(dataset_test))
        d_train = dataset_test[0]
        print(self.tokenizer.decode(d_train["source_ids"], skip_special_tokens=False))
        print(self.tokenizer.decode(d_train["target_ids"], skip_special_tokens=False))

        dataloader = DataLoader(dataset_test, batch_size=self.args.eval_batch_size, num_workers=2)

        self.model.model.eval()
        model = self.model.to("cpu")

        i = 0
        for batch in dataloader:
            i += 1
            outs = model.model.generate(input_ids=batch['source_ids'],
                                        attention_mask=batch['source_mask'])
            predictions = [self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                           .strip() for ids in outs]
            targets = [self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                       for ids in batch["target_ids"]]
            texts = [self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                     for ids in batch["source_ids"]]

            print(texts)
            print(predictions)
            print(targets)

            for i in range(4):
                text = texts[i]
                print(text)
                print("\nActual Entities: %s" % targets[i])
                print("Predicted Entities: %s" % predictions[i])
                print("=====================================================================\n")
            if i == 11:
                break

    def predict_single(self, test_sample):
        self.model.model.eval()
        model = self.model.to("cpu")
        tokenized_inputs = self.tokenizer.encode(test_sample, return_tensors="pt")

        model_output = model.model.generate(input_ids=tokenized_inputs)
        predictions = [self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                       .strip() for ids in model_output]
        texts = self.tokenizer.decode(tokenized_inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        predictions = [pred.strip() for pred in predictions[0].split(';') if ':' in pred]
        predictions = [pred.strip() for pred in predictions if pred.split(':')[0] and pred.split(':')[1]]

        print(texts)
        print(predictions)
        print(set(predictions))


if __name__ == "__main__":
    set_seed(42)
    model_path = "../models/feature-ext.ckpt"
    dataset = load_from_disk("../dataset/entities_ds/ner_dataset")

    args_dict = dict(
        data_dir="../dataset/entities_ds/ner_dataset",
        output_dir="../models/",
        model_name_or_path='t5-small',
        tokenizer_name_or_path='t5-small',
        check_point=model_path,
        max_seq_length=40,
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=1,
        eval_batch_size=1,
        num_train_epochs=3,
        gradient_accumulation_steps=16,
        n_gpu=1,
        early_stop_callback=False,
        fp_16=True,
        opt_level='O1',
        max_grad_norm=1,
        seed=42,
    )

    evaluate_ner = EvaluationNER(args_dict)
    # evaluate_ner.evaluate(dataset)
    # evaluate_ner.save_model()
    evaluate_ner.evaluate_label(dataset)
    evaluate_ner.predict_single("0.1 degree")
    evaluate_ner.predict_single("We want to look at this perovskite to understand its structure. We think 5 seconds of exposure should be sufficient. Theta of 0.2 would be good.")
    evaluate_ner.predict_single("If a material exhibits a spacing between crystal lattice planes d of 6, what does this correspond to in terms of the reciprocal/Fourier space scattering vector q?")
