# Question-Answering-SQUAD
Question Answering is the task of answering questions (typically reading comprehension questions), but abstaining when presented with a question that cannot be answered based on the provided context. The main method we rely on in this project is Transformer. Specifically, we take the DistilBERT model, pretrained on Masked LM and Next Sentence Prediction, add new head for question answering, and train the new model for our task. 
The reason we used pre-trained transformers instead of building a specific deep learning model (LSTM, CNN, etc.) which is suitable for question answering tasks is that we could do a quicker development and we could get better results by using fewer data. 
In fact, we believe that the same transfer learning shift as the one that took place in CV field several years ago would happen to NLP. Rather than training a new network from scratch each time, the lower layers of a trained network with generalized features (the backbone) could be copied and transferred for use in another network with a different task.
To come up with the best model we have experimented with different question heads which differ in the number of layers, activation function and overall structure. 
For the analysis of the results, we measured the F1 score and the EM (Exact Match) score to have the same set of metrics as in the SQuAD paper and we found that our question head led to better results with respect to the Vanilla one present on the Hugging Face library.

## Installation

To quickly use our modified version of DistilBertForQuestionAnswering, clone this repository and install the necessary requirements by running

`pip install -r requirements.txt`

We recommend creating a separate python 3.6 environment. 

## Authors

**Simone Gayed Said** - simone.gayed@studio.unibo.it </br>
**Alex Rossi** - alex.rossi6@studio.unibo.it </br>
**Jia Liang Zhou** - jialiang.zhou@studio.unibo.it
**Hanying Zhang** - hanying.zhang@studio.unibo.it </br>

## Useful Links

**Hugging Face library** - https://huggingface.co/transformers/ </br>
**Our organization on the Hugging Face Hub** - https://huggingface.co/nlpunibo
