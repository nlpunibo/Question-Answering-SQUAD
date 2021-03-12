import argparse
from load import *
from utils import *
from model import *
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer


def main():
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument("path_to_json_file", help="Path to the json file", type=str)
    args = parser.parse_args()

    # Set the seed for reproducibility
    fix_random(seed=42)

    # Load the data
    data_path = Path(args.path_to_json_file).parent
    _ = LoadData(args.path_to_json_file, str(data_path))

    test_data = load_dataset('json', data_files=str(data_path / "test.json"), field='data')

    # Preprocessing the test data

    # Before we can feed those texts to our model, we need to preprocess them.
    # This is done by a Transformers Tokenizer which will(as the name indicates) tokenize
    # the inputs(including converting the tokens to their corresponding IDs in the pretrained
    # vocabulary) and put it in a format the model expects, as well as generate the
    # other inputs that model requires.

    # To do all of this, we instantiate our tokenizer
    # with the AutoTokenizer.from_pretrained method, which will ensure:
    # - we get a tokenizer that corresponds to the model architecture we
    #   want to use,
    # - we download the vocabulary used when pretraining this specific
    #   checkpoint.

    # As model_checkpoint we use the best performing version of our DistilBertForQuestionAnswering which we
    # have uploaded on the HuggingFace Hub
    model_checkpoint = "nlpunibo/distilbert_config3"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # The maximum length of a feature (question and context)
    max_length = 384

    # The authorized overlap between two part of the context when splitting it is needed.
    doc_stride = 128

    # Our model expects padding on the right
    pad_on_right = True

    squad = SQUAD(tokenizer, pad_on_right, max_length, doc_stride)

    # We can download the pretrained model.
    # We use our modified version of the `DistilBertForQuestionAnswering` class.
    # Like with the tokenizer, the `from_pretrained` method will download and cache the model for us.
    model = DistilBertForQuestionAnswering.from_pretrained(model_checkpoint)

    # Tell pytorch to run this model on the GPU.
    if torch.cuda.is_available():
        model.cuda()

    # We instantiate a `Trainer`, that will be used to get the predictions.
    # Note: This is not necessary, but using the Trainer instead of directly the model to get the predictions simplify this operation.
    args = TrainingArguments(
        output_dir= str(data_path / "results"),
        label_names=["start_positions", "end_positions"]
    )
    trainer = Trainer(model, args)

    # The only point left is how to check a given span is inside the context (and not the question) and how to get back the text inside.
    # To do this, we need to add two things to our test features:
    #   - the ID of the example that generated the feature (since each example can generate several features, as seen before);
    #   - the offset mapping that will give us a map from token indices to character positions in the context.
    # That's why we will process the test set with the prepare_validation_features
    print("Preparing the test data:")
    test_features = test_data['train'].map(squad.prepare_validation_features, batched=True,
                                           remove_columns=test_data['train'].column_names)

    # Get final predictions
    with torch.no_grad():
        pred = trainer.predict(test_features)

    # The Trainer hides the columns that are not used by the model (here example_id and offset_mapping which we will need for our post-processing), so we set them back
    test_features.set_format(type=test_features.format["type"],
                             columns=list(test_features.features.keys()))

    # To get the final predictions we can apply our post-processing function to our raw predictions
    final_predictions = dict(squad.postprocess_qa_predictions(test_data['train'], test_features, pred.predictions))

    # Create a new file and save the predictions
    with open(data_path / "results" / "predictions.json", 'w') as file:
        file.write(json.dumps(final_predictions))
        file.close()


if __name__ == '__main__':
    main()
