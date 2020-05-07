---

layout: post
title: Natural Language Processing with Tensorflow (Part 1)
description: creating a simple application for finding positive and nwgative review from IMDB dataset. Notes of Coursera course.
keywords: IMDB, Coursera, NLP, embedding, review, AI, ML, machine learning, deep learning
author: Parikshit Patil
thumbnail: https://pskp-95.github.io/public/images/thumbnail_nlp1.png
---

This blog post is a notes on course **Natural Language Processing with Tensorflow on Coursera**.

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

![img](data:image/png;base64,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)

![img](data:image/png;base64,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)

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

