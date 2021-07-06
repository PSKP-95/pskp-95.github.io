---

layout: post
title: Natural Language Processing with Tensorflow (Part 1)
description: creating a simple application for finding positive and nwgative review from IMDB dataset. Notes of Coursera course.
keywords: IMDB, Coursera, NLP, embedding, review, AI, ML, machine learning, deep learning
author: Parikshit Patil
thumbnail: https://pskp-95.github.io/public/images/thumbnail_nlp1.png
views: 43
---

This blog post is a notes on course **Natural Language Processing with Tensorflow on Coursera**.

<div class="index">
<h2>Table of Contents</h2>
<ul id="myUL">
  <li><a href="#preprocessing">Preprocessing</a></li>
  <ul>
    <li><a href="#encoding-sentences-in-tensorflow--keras">Encoding Sentences in Tensorlflow / Keras</a></li>
    <li><a href="#creating-sequences-from-sentences">Creating Sequences from sentences</a></li>
    <li><a href="#padding">Padding</a></li>
  </ul>
  <li><a href="#lets-learn-with-small-example">Lets Learn with small example</a></li>
  <ul>
    <li><a href="#preprocessing-as-we-done-above">Preprocessing as we done above</a></li>
    <li><a href="#create-model-in-keras">Create Model in Keras</a></li>
    <li><a href="#what-is-embedding">What is embedding</a></li>
  </ul>
</ul>
</div>

## Preprocessing

The most important part of natural language processing is embedding means **How to represent sentences?** for example below if we label every single character in word but here is problem. both words **SILENT** and **LISTEN** has same letters.

<img src="https://pskp-95.github.io/public/images/silent_listen.png">

So another approach, represent words with labels. As given below, **I** &rarr; **001** which is fixed while training and testing.

<img src="https://pskp-95.github.io/public/images/i_love.png">

### Encoding Sentences in Tensorflow / Keras

Let's import packages and libraries required.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  # We will use it later
```

Now lets take some sentences and **tokenize** them.

```python
sentences = [
    'I love my dog',
    'I love my cat so much',
    'You love my dog!'
]

# Create object of Tokenizer and tokenize sentences
tokenizer = Tokenizer(num_words = 100)  # max distinct words first 100 words
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index  # returns dictionary
print(word_index)
```

**output**

<div style="background-color:black;color:white;width:100%;padding-left:10px">{'love': 1, 'my': 2, 'i': 3, 'dog': 4, 'cat': 5, 'so': 6, 'much': 7, 'you': 8}</div>

As you can see,

1. Words are lower-cased
2. **!** mark removed

### Creating Sequences from sentences

Now we need sequence of codes for each sentence so that we can pass these sequences to **neural Network**.

We already created **Tokenizer** instance and passed our sentences to its method **tokenizer.fit_on_texts(sentences)**.

```python
# Let's append new sentence to `sentences`
sentences.append("I love my Tom")

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
```

**output**

<div style="background-color:black;color:white;width:100%;padding-left:10px">[[3, 1, 2, 4], [3, 1, 2, 5, 6, 7], [8, 1, 2, 4], <span style="background-color:white;color:red">[3, 1, 2]]</span></div>

But here we have another problem, **What if word is new to dictionary?**  see last output, our dictionary don't have word **Tom** and tensorflow just ignored it. And always it does same, it ignores these words. But it is unfair and may be we can **add some wild / special value for unseen words.**

Let's start again

```python
# at Tokenizer instance creation add another keyword argument
tokenizer = Tokenizer(num_words=100, oov_token='<OOv>')  # OOV means Out of vocabulary

...... # do same as above
print(word_index)
print(sequence)
```

**output**

<div style="background-color:black;color:white;width:100%;padding-left:10px">{<span style="background-color:white;color:red">'&lt; OOv &gt;': 1,</span>'love': 2, 'my': 3, 'i': 4, 'dog': 5, 'cat': 6, 'so': 7, 'much': 8, 'you': 9} <br> [[4, 2, 3, 5], [4, 2, 3, 6, 7, 8], [9, 2, 3, 5], [4, 2, 3,<span style="background-color:white;color:red"> 1 </span>]]</div>

As you can see, **Tom** is not in corpus and it is asisgned as **&lt;'OOV'&gt;**. But as our dataset increases, possibility of missing word reduces. or We can use **GloVe** or **word2vec** embeddings. *We will see later*.

### Padding

But still here is problem, as we know neural networks works **only with fixed size input**. so we need to **pad these sequences / trim to fixed length**.

Until now we got sequences from our sentences with different length of each sequence. Let's pad them to same length.

```python
padded_sequences = pad_sequences(sequences, padding='post',truncating='post',maxlen=5)
print(padded_sequences)
```

**output**

<div style="background-color:black;color:white;width:100%;padding-left:10px">[[4 2 3 5 <span style="background-color:white;color:red"> 0 </span>] <br>
 [4 2 3 6 7] <span style="background-color:white;color:red"> last 'much' removed </span><br>
 [9 2 3 5 <span style="background-color:white;color:red"> 0 </span>]<br>
 [4 2 3 1 <span style="background-color:white;color:red"> 0 </span>]]</div>

## Lets Learn with Small Example

We will create **positive and negative review analysis** system from **IMDB dataset**. For setting thing, we need to install **tensorflow-dataset** or use **google colab** instead.

<div style="background-color:black;color:white;width:100%;padding-left:10px"><span style="color:green">parikshit@Vostro-5568</span>:~$ pip install -q tensorflow-datasets</div>

Once **tensorflow-datasets** installed, load **IMDB Dataset**.

```python
import tensorflow_datasets as tfds

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
```

This block will download dataset if not found and store in most efficient format **TFrecords**.

Split data for **Training and Testing **. and convert in appropriate format as we need in above example &rarr;**Separate sentences and labels**

```python
import numpy as np

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for sentence, label in train_data:
    training_sentences.append(str(sentence.numpy()))
    training_labels.append(label.numpy())

for sentence, label in test_data:
    testing_sentences.append(str(sentence.numpy()))
    testing_labels.append(label.numpy())

print("Number of training samples: " + str(len(training_sentences)))
print("Number of testing samples: " + str(len(testing_sentences)))
```

**output**

<div style="background-color:black;color:white;width:100%;padding-left:10px">Number of training samples: 25000 <br>
Number of testing samples: 25000</div>

Lets do some **exploratory data analysis**

```python
# print 1st sentence and corrsponding label in traing data
print("Sentence: " + training_sentences[0])
print("Label: " +str(training_labels[0]))
```

**output**

<div style="background-color:black;color:white;width:100%;padding-left:10px">
Sentence: b"This was an absolutely terrible movie. Don't be lured in by Christopher Walken or Michael Ironside. Both are great actors, but this must simply be their worst role in history. Even their great acting could not redeem this movie's ridiculous storyline. This movie is an early nineties US propaganda piece. The most pathetic scenes were those when the Columbian rebels were making their cases for revolutions. Maria Conchita Alonso appeared phony, and her pseudo-love affair with Walken was nothing but a pathetic emotional plug in a movie that was devoid of any real meaning. I am disappointed that there are movies like this, ruining actor's like Christopher Walken's good name. I could barely sit through it." <br>
Label: 0</div>

Our training and testing data labels is in **list** but for training we need to convert it into **numpy array**

```python
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

## check by printing shapes
print("Shape of training labels: " + str(training_labels_final.shape))
print("Shape of testing labels: " + str(testing_labels_final.shape))
```

**output**

<div style="background-color:black;color:white;width:100%;padding-left:10px">Shape of training labels: (25000,)<br>
Shape of testing labels: (25000,)</div>

### Preprocessing as we done above

We need to preprocess data as we done above. Code is same as above.

```python
## define params so that we can change easily
vocab_size = 100000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
pad_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(traing_sentences)
training_padded = pad_sequences(trainining_sequences, maxlen= max_length, padding=pad_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,  padding=pad_type, truncating=trunc_type)
```

We Lets check our training data and testing data shape

```
print("Training Input: " + str(training_sequences.shape))

print("Testing Input: " + str(testing_sequences.shape))
```

**output**

<div style="background-color:black;color:white;width:100%;padding-left:10px">Training Input: (25000, 120)<br>
Testing Input: (25000, 120)</div>

### Create Model in Keras

Keras provide higher level API for tensorflow so it is too easy to create and train neural networks.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6,activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())
```

**output**

<div><pre style="background-color:black;color:white;padding-left:10px">
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 120, 16)           1600000
_________________________________________________________________
flatten (Flatten)            (None, 1920)              0
_________________________________________________________________
dense (Dense)                (None, 6)                 11526
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 7
=================================================================
Total params: 1,611,533<
Trainable params: 1,611,533
Non-trainable params: 0
_________________________________________________________________</pre>
</div>

### Compile and Start Training Model

```python
## Compile Model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

num_epochs = 10

history = model.fit(training_padded,
         training_labels_final,
         epochs=num_epochs,
         validation_data=(testing_padded, testing_labels_final))
```

**output**

<div>
<pre style="background-color:black;color:white;padding-left:10px">
Epoch 1/10
782/782 [==============================] - 8s 10ms/step - loss: 0.5179 - accuracy: 0.7336 - val_loss: 0.3833 - val_accuracy: 0.8250
Epoch 2/10
782/782 [==============================] - 8s 10ms/step - loss: 0.2824 - accuracy: 0.8834 -
...
Epoch 10/10
782/782 [==============================] - 8s 10ms/step - loss: 2.0107e-04 - accuracy: 0.9999 - val_loss: 1.3380 - val_accuracy: 0.7828
</pre>
</div>

### Let's Visualise training-validation loss-accuracy

```python
import matplotlib.pyplot as plt
# accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
```

**output**

![img](/public/images/accuracy_loss.png)

![img](/public/images/loss_nlp.png)

### What is embedding

**tf.keras.layers.embedding(..)** turns positive integers (We encoded in preprocessing stage) into dense vectors of fixed size. **This layer can only be used as first layer in a model**

We will learn about embedding in detail here, until now let's visualise embedding in  **Tensorflow embedding projector**

Find reverse map **index &rarr; Word**

```python
# Reverse map from index to word
inv_map = {v: k for k, v in word_index.items()}
```

Now get weights of embedding layer i.e. first layer. Its shape must be **vocab_size x embedding_dim**

```python
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)
```

**output**

<div style="background-color:black;color:white;width:100%;padding-left:10px">(100000, 16)</div>

Now download these weights and words in .tsv file. Then go to **Tensorflow embedding projector**  and visualise.

```python
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

vocab_size = len(inv_map.keys())

for word_num in range(1, vocab_size):
    word = inv_map[word_num]
    embeddings = weights[word_num]
    out_m.write(word + '\n')
    out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')

out_v.close()
out_m.close()
```

<hr>

### References and Code

[1] [Natural Language Processing in Tensorflow  by Laurence Moroney](https://www.coursera.org/learn/natural-language-processing-tensorflow)

You can find code  <a target="_blank" href="https://colab.research.google.com/drive/1EcqFDKb_fNb6EApjOxifq9QwLJok9WcD" target="_parent"><img style="float:right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

