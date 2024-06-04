"""Suicide_nlp4.ipynb

# Code has been initially started in Google colab, but then transfered to VS Code (locally).

## Fine-tune GPT-2 Model with Tensorflow in Transformers for text generation
We build a model that can be prompted to generate human like Reddit Posts on Suicidal Ideation & Depression. For that, we fine-tune GPT-2 on an unique data set using Tensorflow in Transformers. For fast results, we use TPUs on Google Colab.

We combine the depression & suicidal ideation dataset.

## Motivation

Here, we move to text generation with neural networks. Because our data set is not only extensive but also of high quality, it lends itself perfectly for training a neural network. As before, we make use of the power of transfer learning: We use the English GPT-2 model.

We do:
 * We fine-tune the model using the transformers library
 * We use the Tensorflow instead of PyTorch implementation --> loading Hugging face models with TF as prefix!
 * if possible, we run on TPU
 * Our data set has text comments and their corresponding rating. Hence, we're able to teach our model to generate positive and negative reviews, as requested

### Setup / Data set / cleaning / pre processing

Same as before :)  
We will need to use a [TPU](https://cloud.google.com/tpu/docs/colabs) because of the high computational demand of GPT-2 and the size of the data. While you can get away with using GPUs as well, you won't stand any chance to run this on a CPU.   
Now, let's get rolling:
"""

import os
#!pip install --upgrade pip setuptools
#!pip install tf-keras

import warnings
import re
import random
import transformers
import datasets
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from transformers import GPT2Tokenizer, TFGPT2Model, AutoTokenizer
from datasets import Dataset, load_dataset

pd.options.display.max_colwidth = 6000
pd.options.display.max_rows = 400
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")
# Log Level and suppress extensive tf warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print(tf.__version__)
print(transformers.__version__)

PATH_BASE = "/Users/samanthaweber/NLP_Suicide"

# For files in Google Drive
# Store the dataset in your Google drive.
CURR_PATH = PATH_BASE


strategy = 1

# Download and extract data
data = pd.read_csv(CURR_PATH + "/combined-GPT-finetuning.csv")

print(data.columns)

# mark at risk posts
data["label"] = np.where(
    data["label"] == 1, "<|at_risk|>", "<|no_risk|>"
)
data.head(3)

"""As we still have the previous labels, we can instead of depression/suicidality yes/no look at it as "at risk" yes/no."""

data.head(3)

"""Here's quick glimpse at the composition of our reviews:"""

data["label"].value_counts()

"""In total we have 5447 reviews. Quite balanced, but probably not enough for fine-tuning.  
Following, we pre process the comments. We do so by removing irregular characters from the text and making sure we use consistent spacing - as done before:
"""

# Commented out IPython magic to ensure Python compatibility.
# Preprocessing
def clean_text(text):
     """
     - remove any html tags (< /br> often found)
     - Keep only ASCII + Latin chars, digits and whitespaces
     - pad punctuation chars with whitespace
     - convert all whitespaces (tabs etc.) to single wspace
     """
     RE_PUNCTUATION = re.compile("([!?.,;-])")
     RE_TAGS = re.compile(r"<[^>]+>")
     RE_ASCII = re.compile(r"[^A-Za-zÀ-ž,.!?0-9 ]", re.IGNORECASE)
     RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
     text = re.sub(RE_TAGS, " ", text)
     text = re.sub(RE_ASCII, " ", text)
     text = re.sub(RE_PUNCTUATION, r" \1 ", text)
     text = re.sub(RE_WSPACE, " ", text)
     return text
 
 
 # Clean Comments. Only keep long enough
data["text_clean"] = data.loc[data["text"].str.len() > 1, "text"]
data["text_clean"] = data["text_clean"].map(
     lambda x: clean_text(x) if isinstance(x, str) else x
)

# Drop Missing and save to file
data = data.dropna(axis="index", subset=["text_clean"]).reset_index(drop=True)
# add label as first word of comment
data["text_clean"] = data["label"] + " " + data["text_clean"]
data = data[["text_clean"]]
data.columns = ["text"]
#data.to_csv(PATH_BASE + "/suicide_clean_rating.csv", index=False)
print(data.head(2))

"""That'll be the foundation for creating the input to our GPT-2 model. We didn't have to do too much with our text.

### Create Model Inputs
Following, we use the [datasets](https://huggingface.co/docs/datasets/) library to convert the pandas dataframe to a `Dataset` object. This object has some helpful properties when working with the transformers library:
"""

# Read data from file and load as dataset
#data = pd.read_csv(PATH_BASE + "/suicide_clean_rating.csv")
data = Dataset.from_pandas(data)
print(data)

"""We use the pre trained tokenizer for German that comes with the model. As we've introduced tokens for negative and positive reviews, we add them to the tokenizer's vocabulary using `add_tokens`. We also add a new token for padding:"""

MAX_TOKENS = 128
POS_TOKEN = "<|no_risk|>"
NEG_TOKEN = "<|at_risk|>"
BOS_TOKENS = [NEG_TOKEN, POS_TOKEN]
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"

from transformers import DistilBertTokenizer, AutoModel, GPT2TokenizerFast
# this will download and initialize the pre trained tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained(
    'gpt2',
    #'distilbert/distilbert-base-uncased',
    eos_token=EOS_TOKEN,
    pad_token=PAD_TOKEN,
    max_length=MAX_TOKENS,
    is_split_into_words=True,
    return_tensors="tf",
)
tokenizer.add_tokens(BOS_TOKENS, special_tokens=True)


"""Coming up, we prepare the model input. For that, we have to do several tasks:
1. We add our end of sentence (eos) token, `<|endoftext|>` at the end of each comment
2. We tokenize each comment using the GPT-2 tokenizer
3. While doing so, we add padding (using `<|pad|>` as a token) / or truncate each comment so that is has exactly `MAX_TOKENS` tokens
4. The tokenizer then also takes care of converting the tokens to a numeric representation `input_ids` (each token corresponds to a specific numeric id)
5. Next, we need to create `labels` as an input as well. These are actually the same as the shifted `input_ids`, but we replace ids corresponding to our padding token with `-100`
6. Finally, the tokenizer also creates the `attention_mask`. This is just a vector consisting of `1` for all relevant elements in `input_ids` and `0` for all padding tokens --> this makes that Padding tokens will be ignored during the learning, but the input tensor is still of same dimension.

"""

# Commented out IPython magic to ensure Python compatibility.

 
output = {}
# texts to numeric vectors of MAX_TOKENSue
def tokenize_function(examples, tokenizer=tokenizer):
    # Add start and end token to each comment
    examples = [ex + EOS_TOKEN for ex in examples["text"]]
    # tokenizer created input_ids and attention_mask as output
    output = tokenizer(
        examples,
        add_special_tokens=True,  # Only adds pad not eos and bos
        max_length=MAX_TOKENS,
        truncation=True,
        pad_to_max_length=True,
        return_tensors='tf',
    )
    # shift labels for next token prediction
    # set padding token labels to -100 which is ignored in loss computation
    output["labels"] = [x[1:] for x in output["input_ids"]]
    output["labels"] = [
        [0 if x == tokenizer.pad_token_id else x for x in y]
        for y in output["labels"]
    ]
    # truncate input ids and attention mask to account for label shift
    output["input_ids"] = [x[:-1] for x in output["input_ids"]]
    output["attention_mask"] = [x[:-1] for x in output["attention_mask"]]
    return output
 
 
data = data.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["text"],
    load_from_cache_file=True,
)

print(data)
print(data[3]['input_ids'])
print(data[3]['attention_mask'])
print(data[3]['labels']) # The labels column has the tokens shifted one to the right in comparison to input_ids, as it represents the next token that should be predicted. 

"""The `map` method applies `tokenize_function` to our data set. By default, `batched=True` applies the function to a batch of 10k rows and using `num_proc` we can do this in parallel. We use a `MAX_TOKENS` value of 128. This covers about 80% of all reviews in our data set without truncation. Increasing this is costly in terms of additional computation time, so this is a good compromise.  

What to do after we've applied `tokenizer` to our batches:
In particular, the shifting of `input_ids` and `labels` happens inside the model during training when using the PyTorch model but not in the Tensorflow version. Thus, we need to explicitly do it beforehand. But why do we need the shifting at all? Well, we train GPT-2 on the task of causal language modeling. Basically, given a sequence of words the model learns to guess the next word. However, the model uses ids not words. So, given a sequence of `input_ids` what will be the next `input_id`? For computing the loss on this task, the model compares its predicted output with a label. Hence, `labels` must be the shifted `input_ids`.  
Moreover, we don't want the model to train on or predict padding tokens. This is where the [attention mask](https://huggingface.co/transformers/glossary.html#attention-mask) comes into play. But this must be also be taken into account during loss computation. The transformer implementation of GPT-2 does this internally by ignoring all labels that are `-100`. Hence, we adapt our `labels` accordingly.  


Now, we create a split for training and testing and convert the data to the appropriate format for Tensorflow before we can start building our model:
"""

#'''
# Load Inputs and create test and train split
#from datasets import load_from_disk
from datasets import load_from_disk
from sklearn.model_selection import train_test_split # type: ignore

data.save_to_disk(PATH_BASE + "/suicide_tokenized_128_ratings")
data = datasets.load_from_disk(PATH_BASE + "/suicide_tokenized_128_ratings")
print(data)
data.set_format(type="python", columns=["input_ids", "attention_mask", "labels"])
data = data.train_test_split(
    test_size=0.20, shuffle=True, seed=1, load_from_cache_file=True
)
print(data)


# prepare for use in tensorflow
train_tensor_inputs = tf.convert_to_tensor(data["train"]["input_ids"])
train_tensor_labels = tf.convert_to_tensor(data["train"]["labels"])
train_tensor_mask = tf.convert_to_tensor(data["train"]["attention_mask"])
train = tf.data.Dataset.from_tensor_slices(
    (
        {"input_ids": train_tensor_inputs, "attention_mask": train_tensor_mask},
        train_tensor_labels,
    )
)

test_tensor_inputs = tf.convert_to_tensor(data["test"]["input_ids"])
test_tensor_labels = tf.convert_to_tensor(data["test"]["labels"])
test_tensor_mask = tf.convert_to_tensor(data["test"]["attention_mask"])
test = tf.data.Dataset.from_tensor_slices(
    (
        {"input_ids": test_tensor_inputs, "attention_mask": test_tensor_mask},
        test_tensor_labels,
    )
)

print(test)
"""This concludes the data preparation.

### Build and train GPT-2 Model

Next, we can start defining our model architecture:
"""

# Model params
BATCH_SIZE_PER_REPLICA = 128
EPOCHS = 6
INITAL_LEARNING_RATE = 0.001

BATCH_SIZE = BATCH_SIZE_PER_REPLICA
BUFFER_SIZE = len(train)

# prepare data for consumption
train_ds = (
    train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
)
test_ds = test.batch(BATCH_SIZE, drop_remainder=True)

print(train_ds)


# Drecreasing learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    INITAL_LEARNING_RATE,
    decay_steps=500,
    decay_rate=0.7,
    staircase=True)


from transformers import TFGPT2LMHeadModel
#model = AutoModelForCausalLM.from_pretrained('gpt2')
# initialize model, use_cache=False important! else wrong shape at loss cal
model =TFGPT2LMHeadModel.from_pretrained(
    'gpt2',
    #from_pt=False,
    use_cache=False,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model.resize_token_embeddings(len(tokenizer))
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
#model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])
model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer])
model.summary()

"""
Before we start training, let's define some callbacks that will be used during training:
"""

# Stop training when validation acc starts dropping
# Save checkpoint of model after each period
now = datetime.now().strftime("%Y-%m-%d_%H%M")
# Create callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", verbose=1, patience=1, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        PATH_BASE + "/models/" + now + "_GPT2-Model_{epoch:02d}_{val_loss:.4f}.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
]

"""
We will stop model training prematurely, if the the validation loss does not improve after an epoch. Also, we make sure to save model checkpoints after each epoch so that we can resume training later on.  
Now, we're all set for training. Let's start fine-tuning our model:
"""



# # Train Model
steps_per_epoch = int(BUFFER_SIZE // BATCH_SIZE)
print(
    f"Model Params:\nbatch_size: {BATCH_SIZE}\nEpochs: {EPOCHS}\n"
    f"Step p. Epoch: {steps_per_epoch}\n"
    f"Initial Learning rate: {INITAL_LEARNING_RATE}"
)

hist = model.fit(
    train_ds,
    validation_data=test_ds,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

loss = pd.DataFrame(
    {"train loss": hist.history["loss"], "test loss": hist.history["val_loss"]}
).melt()
loss["epoch"] = loss.groupby("variable").cumcount() + 1
sns.lineplot(x="epoch", y="value", hue="variable", data=loss).set(
    title="Model loss",
    ylabel="",
    xticks=range(1, loss["epoch"].max() + 1),
    xticklabels=loss["epoch"].unique(),
);


"""After training for three epochs and about 45 minutes, the validation loss is around 2.8273 and seems to be flattening out. I'm sure, with some longer training times and adaptation of the hyperparameters we could improve this further.

### Text generation

Let's see how our training results translate to the quality of text generation. Using the [pipelines](https://huggingface.co/transformers/main_classes/pipelines.html) class in transformers is straight forward for text generation:
We create now some "Reddit Posts" that are either representing subjects at risk or not at risk for suicide.
"""

# Restored Trained Model weights
#model.load_weights(PATH_BASE + "/models/2024-05-12_1342_GPT2-Model_06_2.9055.h5")

from transformers import pipeline

reddit_post = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

"""We call our `reddit_post` object using the token we defined for "at risk" as a prompt. Remember that we have trained our model to implicitly differentiate between at risk and not at risk using the appropriate tokens in the inputs. 
Consequently, the model hopefully learned to create reviews with different sentiments depending on the prompt we feed it. We wish to create outputs that are at most 250 tokens long,
(the model stops either when it generates an `<|endoftext|>` token or after `max_length` tokens) . Lastly, we ask for six different example reviews:"""

gen_risk = reddit_post("<|at_risk|>", max_length=250, num_return_sequences=6)
df=pd.DataFrame(gen_risk[:4])
print(df)

""" Was it enough to fine-tune our model for creating depression/suicide related posts? Let's check it out:"""

gen_norisk = reddit_post("<|no_risk|>", max_length=250, num_return_sequences=6)
df = pd.DataFrame(gen_norisk[:4])
print(df)
