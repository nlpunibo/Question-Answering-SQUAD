# Question-Answering-SQUAD
Question answering is an important task based on which intelligence of NLP systems and AI in general can be judged. A QA system is given a short paragraph or *context* about some topic and is asked some questions based on the passage. The answers to these questions are spans of the context, that is they are directly available in the passage. To train such models, we use the [SQUAD](https://arxiv.org/abs/1606.05250) dataset.

The main method we rely on in this project is Transformer. Specifically, we take the DistilBERT model, pretrained on Masked LM and Next Sentence Prediction, add new head for question answering, and train the new model for our task. 

The reason we used pre-trained transformers instead of building a specific deep learning model (LSTM, CNN, etc.) which is suitable for question answering tasks is that we could do a quicker development and we could get better results by using fewer data. 
In fact, we believe that the same transfer learning shift as the one that took place in CV field several years ago would happen to NLP. Rather than training a new network from scratch each time, the lower layers of a trained network with generalized features (the backbone) could be copied and transferred for use in another network with a different task.

To come up with the best model we have experimented with different question heads which differ in the number of layers, activation function and overall structure. 
For the analysis of the results, we measured the F1 score and the EM (Exact Match) score to have the same set of metrics as in the SQuAD paper and we found that our question head led to better results with respect to the Vanilla one present on the Hugging Face library.

## Installation

To quickly use our modified version of DistilBertForQuestionAnswering, clone this repository and install the necessary requirements by running

`pip install -r requirements.txt`

We recommend creating a separate python 3.6 environment. 

## Usage

To run the two scripts `train.py` and the `eval.py`, you just need to launch respectively:
- `python3 train.py [path_to_json_file]`
- `python3 get_predictions.py [path_to_json_file]`

Where `path_to_json_file` is the path to the json file which, in the `train.py` case, will be used to train our custom DistilBertForQuestionAnswering model whereas, in the `get_predictions.py` case, it will be used to compute and save another JSON file with the following format

```json
{
    "question_id": "textual answer",
    ...
}
```
 - `python3 evaluate.py [path_to_json_file] [prediction_file]`: given the path to the same testing json file used in the `get_predictions.py` script and the json file produced by the script itself, prints to the standard output a dictionary of metrics such as the `F1` and `Exact Match` scores, which can be used to assess the performance of a trained model as done in the official SQuAD competition

The two Colab notebooks `DistilbertQA_train.ipynb` and `DistilbertQA_eval.ipynb` provide more comments and useful plots w.r.t the python scripts. If you want to use them make sure to have a Google Drive folder with the json files you want to use and to change in the notebooks the `FOLDER_NAME` and `JSON_TEST_FILE` text fields.

## Recommendations

We strongly reccomend you to use a GPU for running the `train.py` and the `get_predictions.py` scripts. To do so you can use the Nvidia graphic card of your machine, if it has one. In this case make sure that you have all the prerequisites (https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows) and to have installed the pytorch version for the CUDA platform (https://pytorch.org/).

If you don't have an Nvidia GPU at your disposal don't worry we have created for you two Colab Notebooks `DistilbertQA_train.ipynb` and the `DistilbertQA_eval.ipynb`. Colab is a hosted Jupyter notebook service that requires no setup to use, while providing free access to computing resources including GPUs! You will not have to install anything, just navigate to Edit→Notebook Settings, and make sure that GPU is selected as Hardware Accelerator.

## Authors

**Simone Gayed Said** - simone.gayed@studio.unibo.it </br>
**Alex Rossi** - alex.rossi6@studio.unibo.it </br>
**Jia Liang Zhou** - jialiang.zhou@studio.unibo.it </br>
**Hanying Zhang** - hanying.zhang@studio.unibo.it

## Useful Links

**Hugging Face library** - https://huggingface.co/transformers/ </br>
**Our organization on the Hugging Face Hub** - https://huggingface.co/nlpunibo
