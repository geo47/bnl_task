import json
import re
import time

import pandas as pd
from datasets import load_dataset, DatasetDict, load_from_disk
from nltk import word_tokenize


class DataPipeline:
    def __init__(self):
        self.ner_dataset_path = "entities_ds/ner_dataset"
        self.entities_dataset_path = "entities_ds/entities_dataset.jsonl"

    def dump_jsonl(self, data, output_path, append=False):
        """
        Write list of objects to a JSON lines file.
        :param data: python dictionary data to dump in a jsonl file
        :param output_path: path to the output jsonl file
        :param append: append record to the end of file or create a new file
        :return: None
        """
        mode = 'a+' if append else 'w'
        with open(output_path, mode, encoding='utf-8') as f:
            for line in data:
                json_record = json.dumps(line, ensure_ascii=False)
                f.write(json_record + '\n')
        print('Wrote {} records to {}'.format(len(data), output_path))

    def extract_entity_dataset(self, csv_file_path):

        dataset = []
        file_count = 0
        for task_desc_dataset in csv_file_path:
            file_count += 1
            print("Processing file {}. {}".format(file_count, task_desc_dataset + ".csv"))

            task_dataset = pd.read_csv(task_desc_dataset + ".csv", header=0)
            print(task_dataset.head())

            for idx, row in task_dataset.iterrows():

                ner_data_dict = {"text": "", "tokens": [], "spans": []}

                dialogue = row[0]
                """ Extracting entities from dialogue. """
                entities = re.findall('\(.*?\]', dialogue)

                """" Cleaning dialogue text and tokenizing """
                dialogue = re.sub("[\[].*?[\]]", "", dialogue)
                dialogue = re.sub(r"[\(\)]", "", dialogue)
                ner_data_dict["text"] = dialogue
                ner_data_dict["tokens"] = word_tokenize(dialogue)

                """ Extracting entities and task_no """
                unique_entities = set()
                unique_entities.add(str("task_no: " + str(row[1])))

                if entities:
                    for entity in entities:
                        ent_val = re.findall('\(.*?\)', entity)[0].strip()[1:-1]
                        ent_type = re.findall('\[.*?\]', entity)[0].strip()[1:-1]
                        unique_entities.add(str(ent_type + ": " + ent_val))

                """ Adding entities labels to dataset dict """
                if unique_entities:
                    for unique_entity in unique_entities:
                        ner_data_dict["spans"].append(unique_entity)

                """ Appending extracted data dict to final dataset list """
                dataset.append(ner_data_dict)

                """ Empty out data dict to refill in the next iteration """
                ner_data_dict = {"text": "", "tokens": [], "spans": []}

        """ Writing dataset to JSON file """
        self.dump_jsonl(dataset, self.entities_dataset_path)
        time.sleep(1)

    def make_dataset(self):
        """
        Making training dataset from prepared jsonl file
        :return:
        """
        dataset = load_dataset('json', data_files=self.entities_dataset_path).shuffle(seed=42)

        train_test_ds = dataset['train'].train_test_split(test_size=0.1)
        test_valid_ds = train_test_ds['test'].train_test_split(test_size=0.5)

        train_test_valid_dataset = DatasetDict({
            'train': dataset['train'],
            'test': test_valid_ds['train'],
            'valid': test_valid_ds['test']})

        train_test_valid_dataset.save_to_disk(self.ner_dataset_path)


if __name__ == "__main__":
    files = ["raw_files/task_dataset"]

    data_pipeline = DataPipeline()
    data_pipeline.extract_entity_dataset(files)
    data_pipeline.make_dataset()

    dataset = load_from_disk("entities_ds/ner_dataset")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    dev_dataset = dataset["valid"]

    n_train_examples = len(train_dataset)
    n_test_examples = len(test_dataset)
    n_valid_examples = len(dev_dataset)

    print("num train samples: {}".format(n_train_examples))
    print("num test samples: {}".format(n_test_examples))
    print("num valid samples: {}".format(n_valid_examples))

    print(train_dataset[0])
    print(test_dataset[0])
    print(dev_dataset[0])

