import string
from numpy import array
from pickle import load, dump, HIGHEST_PROTOCOL
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import keras
#from keras.backend.tensorflow_backend import set_session
from tensorflow.python.keras.backend import set_session 
import sys, time, os, warnings 
warnings.filterwarnings("ignore")
import re
import tensorflow as tf
import numpy as np
import pandas as pd 
from PIL import Image
import pickle
from collections import Counter
from os import listdir
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Concatenate
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import load_img, img_to_array
from sklearn.utils import shuffle
from keras.applications.vgg16 import VGG16, preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from encoder import Encoder
from decoder import Decoder
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from SWINblock import SwinTransformer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, TensorSpec, function, int64
from keras.losses import sparse_categorical_crossentropy
from transformer import TransformerModel
from time import *
from tensorflow import cast, int32, float32
import tensorflow as tf

from tqdm import tqdm
from tqdm import trange



# Load file into memory
def load_dataset(filename):
    file = open(filename, 'r')
    content = file.read()
    file.close()
    return content

# To remove punctuations
def remove_punctuations(orig_txt):
    punctuation_removed = orig_txt.translate(string.punctuation)
    return punctuation_removed

# To remove single characters 
def remove_single_characters(text):
    len_greater_than_1 = ""
    for word in text.split():
        if len(word) > 1:
            len_greater_than_1 += " " + word
    return len_greater_than_1

# To remove numeric values
def remove_num(text, print_tf=False):
    text_no_num = ""
    for word in text.split():
        isalpha = word.isalpha()
        if print_tf:
            print("   {:10} : {:}".format(word,isalpha))
        if isalpha:
            text_no_num += " " + word
        return text_no_num

# Test to load some images alongside its 5 corresponding captions
images_dir = 'Dataset/Flicker8k_Dataset'
images = listdir(images_dir)

captions_dir = 'Dataset/Flickr8k_text/Flickr8k.token.txt'

print("The number of jpg flies in Flicker8k: {}".format(len(images)))

text = load_dataset(captions_dir)

#Make a dataframe out of raw text
def build_dataset(text):
    data_frame = []
    for sentences in text.split('\n'):
        splitted = sentences.split('\t')
        if len(splitted) == 1:
            continue
        w = splitted[0].split("#")
        data_frame.append(w + [splitted[1].lower()])
    return data_frame  
      
data_frame = build_dataset(text)
# print(len(data_frame))
# print(data_frame[:10])

data = pd.DataFrame(data_frame,columns=["filename","index","caption"])

data = data.reindex(columns=['index','filename','caption'])

#print(data)

# If any filename dosn't have .jpg extension at last then mark it as Invalid filename
def invalid_filename_check(data):
  for filenames in data["filename"]:
    found = re.search("(.(jpg)$)", filenames)
    if (found):
        pass
    else:
        print("Error file: {}".format(filenames))

print(invalid_filename_check(data))  

data[data['filename'] == "2258277193_586949ec62.jpg.1"]

data = data[data['filename'] != '2258277193_586949ec62.jpg.1']
print(data.shape)

def utility_counter(data):
    filenames_unique = np.unique(data.filename.values)
    print("The number of unique filenames : {}".format(len(filenames_unique)))

    count_dict = Counter(data.filename.values)
    # print("Confirming that all the keys have count value as 5")
    # print(count_dict)

    print("The number of captions per image")
    count = Counter(Counter(data.filename.values).values())
    print(count)
    return filenames_unique

unique_filenames = utility_counter(data)

#Plot the images along with the corresponding 5 captions for better data visualization
#Insert reference url here
def data_show(data):
    pic_count = 5
    pixels_count = 384
    target_size = (pixels_count,pixels_count,3)

    count = 1
    fig = plt.figure(figsize=(10,20))
    for jpgfnm in unique_filenames[20:25]:
        filename = images_dir + '/' + jpgfnm
        captions = list(data["caption"].loc[data["filename"]==jpgfnm].values)
        image_load = load_img(filename, target_size=target_size)

        ax = fig.add_subplot(pic_count,2,count,xticks=[],yticks=[])
        ax.imshow(image_load)
        count += 1

        ax = fig.add_subplot(pic_count,2,count)
        plt.axis('off')
        ax.plot()
        ax.set_xlim(0,1)
        ax.set_ylim(0,len(captions))
        for i, caption in enumerate(captions):
            ax.text(0,i,caption,fontsize=20)
        count +=1
    plt.show()        

# data_show(data)

#Create vocabulary
def make_vocab(data):
    vocab = []
    for captions in data.caption.values:
        vocab.extend(captions.split())
    print("Vocabulary Size : {}".format(len(set(vocab))))
    return vocab   

vocabulary = make_vocab(data)

#Find the frequency of words in the dataset
def word_count(data,vocabulary):
    count = Counter(vocabulary)
    append_1 = []
    append_2 = []
    for i in count.keys():
        append_1.append(i)
    for j in count.values():
        append_2.append(j)
    data = {"word":append_1, "count":append_2}
    dfword = pd.DataFrame(data)
    dfword = dfword.sort_values(by='count', ascending=False)
    dfword = dfword.reset_index()[["word","count"]]
    return(dfword)  

df_word_count = word_count(data,vocabulary)   

def text_clean(text_original):
    text = remove_punctuations(text_original)
    text = remove_single_characters(text)
    text = remove_num(text)
    return(text)
    
# for i, caption in enumerate(data.caption.values):
#     newcaption = text_clean(caption)
#     data["caption"].iloc[i] = newcaption

#clean_vocabulary = make_vocab(data)

#Pre process images and captions

#Function to set the path for each image
def image_preprocessor(data):
    vector_all_images = []

    for filename in data["filename"]:
        full_image_path = images_dir+"/"+filename
        vector_all_images.append(full_image_path)
    return vector_all_images

vector_all_images = image_preprocessor(data)
#print(vector_all_images[:10])

def caption_preprocessor(data):
    final_captions = []

    for caption in data["caption"].astype(str):
        caption = '<start> ' + caption + ' <end>'
        final_captions.append(caption)

    return final_captions

final_captions = caption_preprocessor(data)

print(final_captions[:10])

#Resize the image to 224*224*3
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image,(384,384))
    image = tf.image.per_image_standardization(image)
    image = preprocess_input(image)
    #print("sdfdsf", image)
    return image, image_path

print("Total Images : " + str(len(vector_all_images)))
print("Total Captions : " + str(len(final_captions)))

def data_limiter(num,total_captions,all_img_name_vector):
  # Shuffle captions and image_names together
  train_captions, img_name_vector = shuffle(total_captions,all_img_name_vector,random_state=1)
  train_captions = train_captions[:num]
  img_name_vector = img_name_vector[:num]
  return train_captions,img_name_vector

train_captions,img_name_vector = data_limiter(40000,final_captions,vector_all_images)

print("Total Captions = {0} , Total images = {1}".format(len(train_captions),len(img_name_vector)))

encoder_train = sorted(set(vector_all_images))
image_dataset = tf.data.Dataset.from_tensor_slices(encoder_train)
image_dataset = image_dataset.map(load_image).batch(1)

print("Image", image_dataset)

def tokenize_caption(top_k,train_captions):
  # Choose the top 5000 words from the vocabulary
  tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,oov_token="",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
  # oov_token: if given, it will be added to word_index and used to replace out-of-vocabulary words during text_to_sequence calls
  
  tokenizer.fit_on_texts(train_captions)
  train_seqs = tokenizer.texts_to_sequences(train_captions)

  # Map '' to '0'
  tokenizer.word_index[''] = 0
  tokenizer.index_word[0] = ''

  # Create the tokenized vectors
  train_seqs = tokenizer.texts_to_sequences(train_captions)
  return train_seqs, tokenizer

def save_tokenizer(tokenizer, name):
        with open(name + '_tokenizer.pkl', 'wb') as handle:
            dump(tokenizer, handle, protocol=HIGHEST_PROTOCOL)


train_seqs , tokenizer = tokenize_caption(5000,train_captions)

save_tokenizer(tokenizer, 'dec')

print(train_captions[:5])

print(train_seqs[:3])

tokenizer.oov_token

#print(tokenizer.index_word)

#Pad the sequences to the maximum length of the captions
# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)
# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)

# Find the minimum length of any caption in our dataset
def calc_min_length(tensor):
    return min(len(t) for t in tensor)
# Calculates the min_length, which is used to store the attention weights
min_length = calc_min_length(train_seqs)

print('Max Length of any caption : Min Length of any caption = '+ str(max_length) +" : "+str(min_length))

def padding_train_sequences(train_seqs,max_length,padding_type):
  
  cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding=padding_type,maxlen=max_length)
  print("train seqs shape: ", len(cap_vector[0]), cap_vector[0])
  return cap_vector
           
padded_caption_vector = padding_train_sequences(train_seqs,max_length,'post')
print(padded_caption_vector.shape)

img_name_train, img_name_test, caption_train, caption_test = train_test_split(img_name_vector,padded_caption_vector,test_size=0.2,random_state=0)
print("Training Data : X = {0},Y = {1}".format(len(img_name_train), len(caption_train)))
print("Test Data : X = {0},Y = {1}".format(len(img_name_test), len(caption_test)))

#Convert the whole dataset into .npy format
batch_size = 2
buffer_size = 10



#for img, path in tqdm(image_dataset):

  # batch_features = img
  # batch_features = tf.transpose(batch_features, perm=[0, 3, 1, 2])
   # batch_features = tf.reshape(batch_features,(batch_features.shape[0], -1, batch_features.shape[3]))
   # print("BF Shape", batch_features.shape)
  # for bf, p in zip(batch_features, path):
   # path_of_feature = p.numpy().decode("utf-8")
  #  np.save(path_of_feature, bf.numpy())

def load_npy(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  # print("Image Tensor", tf.shape(img_tensor))
  return img_tensor, cap

def create_dataset(img_name_train,caption_train):
  dataset = tf.data.Dataset.from_tensor_slices((img_name_train, caption_train))
  print(dataset)
  # Use map to load the numpy files in parallel
  dataset = dataset.map(lambda item1, item2: tf.numpy_function(load_npy, [item1, item2], [tf.float32, tf.int32]))

  # Shuffle and batch
  dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  print(dataset)
  return dataset

def ddd(img_dir, cap_dir):
	images = listdir(images_dir)
	text = load_dataset(captions_dir)
	data_frame = build_dataset(text)

	data[data['filename'] == "2258277193_586949ec62.jpg.1"]
	data = data[data['filename'] != '2258277193_586949ec62.jpg.1']
	unique_filenames = utility_counter(data)

	vocabulary = make_vocab(data)

	df_word_count = word_count(data,vocabulary)

	vector_all_images = image_preprocessor(data)

	final_captions = caption_preprocessor(data)

	train_captions,img_name_vector = data_limiter(40000,final_captions,vector_all_images)

	encoder_train = sorted(set(vector_all_images))
	image_dataset = tf.data.Dataset.from_tensor_slices(encoder_train)
	image_dataset = image_dataset.map(load_image).batch(1)

	train_seqs , tokenizer = tokenize_caption(5000,train_captions)

	tokenizer.oov_token

	max_length = calc_max_length(train_seqs)

	min_length = calc_min_length(train_seqs)

	img_name_train, img_name_test, caption_train, caption_test = train_test_split(img_name_vector,padded_caption_vector,test_size=0.2,random_state=0)
	print("Training Data : X = {0},Y = {1}".format(len(img_name_train), len(caption_train)))
	print("Test Data : X = {0},Y = {1}".format(len(img_name_test), len(caption_test)))
	
	train_dataset = create_dataset(img_name_train,caption_train)
	test_dataset = create_dataset(img_name_test,caption_test)

	return train_dataset, test_dataset


# Creating train and test dataset
train_dataset = create_dataset(img_name_train,caption_train)
test_dataset = create_dataset(img_name_test,caption_test)

# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack
 
# Define the training parameters
epochs = 10
batch_size = 2
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-5
dropout_rate = 0.1
 

from numpy import random

 
# Implementing a learning rate scheduler
class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)
 
        self.d_model = cast(d_model, float32)
        self.warmup_steps = warmup_steps
 
    def __call__(self, step_num):
 
        # Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)
 
        return (self.d_model ** -0.5) * math.minimum(arg1, arg2)
 
 
# Instantiate an Adam optimizer
optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)
 
# Create model
dec_vocab_size = 8918
dec_seq_length = 39
enc_vocab_size = 8918
enc_seq_length = 39

training_model = TransformerModel(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
 
 
# Defining the loss function
#@function
def loss_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of loss
    padding_mask = math.logical_not(equal(target, 0))
    padding_mask = cast(padding_mask, float32)
 
    # Compute a sparse categorical cross-entropy loss on the unmasked values
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask
 
    # Compute the mean loss over the unmasked values
    return reduce_sum(loss) / reduce_sum(padding_mask)
 
 
# Defining the accuracy function
#@function
def accuracy_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of accuracy
    padding_mask = math.logical_not(equal(target, 0))
 
    # Find equal prediction and target values, and apply the padding mask
    maxpred = cast(argmax(prediction, axis=2),int32)
    accuracy = equal(target, maxpred)
    accuracy = math.logical_and(padding_mask, accuracy)
 
    # Cast the True/False values to 32-bit-precision floating-point numbers
    padding_mask = cast(padding_mask, float32)
    accuracy = cast(accuracy, float32)
 
    # Compute the mean accuracy over the unmasked values
    return reduce_sum(accuracy) / reduce_sum(padding_mask)
 
 
# Include metrics monitoring
train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')
 
# Create a checkpoint object and manager to manage multiple checkpoints
ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)
 
# Speeding up the training process
#@function
def train_step(encoder_input, decoder_input, decoder_output):
    with GradientTape() as tape:
 
        # Run the forward pass of the model to generate a prediction
        prediction = training_model(encoder_input, decoder_input, training=True)
        #print(prediction.shape)
 
        # Compute the training loss
        loss = loss_fcn(decoder_output, prediction)
        # Compute the training accuracy
        accuracy = accuracy_fcn(decoder_output, prediction)
 
    # Retrieve gradients of the trainable variables with respect to the training loss
    gradients = tape.gradient(loss, training_model.trainable_weights)
 
    # Update the values of the trainable variables by gradient descent
    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))
 
    train_loss(loss)
    train_accuracy(accuracy)
 
#outer = tqdm(total=100, desc='Epoch', position=0)
pbar = tqdm(enumerate(train_dataset))
for epoch in (range(epochs)):
 
    train_loss.reset_states()
    train_accuracy.reset_states()
 
    #print("\nStart of epoch %d" % (epoch + 1))
 
    #inner = tqdm(batch_size, desc='Batch', position=1)
    # Iterate over the dataset batches
    for (step, (train_batchX, train_batchY)) in pbar:


        # print(step, train_batchX.shape, train_batchY.shape)
        train_batchX = tf.divide(train_batchX, 255.0)
        # print("Train Batch XXXXXXXXX", train_batchX)
        encoder_input = train_batchX
 
        # Define the encoder and decoder inputs, and the decoder output
        #encoder_input = train_batchX[:, 1:]
        # train_batchY = cast(tf.convert_to_tensor((random.random([64,38])),int32),dtype=int32)
        decoder_input = cast(train_batchY[:, :-1], int32)
        # print(f" decoder input shape: {decoder_input.shape}")
        # encoder_input = cast(tf.convert_to_tensor((random.random([64,3,384,384])),float32),dtype=float32)
        decoder_output = cast(train_batchY[:, 1:], int32)

        
        


 
        train_step(encoder_input, decoder_input, decoder_output)
        pbar.set_postfix({'Epoch, Step, Loss, Accuracy ': [epoch + 1,step,train_loss.result().numpy(),train_accuracy.result().numpy() ]})
        #pbar.set_postfix({f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}'})
 
        if step % 50 == 0:
            save_path = ckpt_manager.save()
            print("Saved checkpoint at epoch %d" % (epoch + 1))
            training_model.save('saved_model/my_model')
            #training_model.save_weights("weights/wghts" + str(epoch + 1) + ".ckpt")
            #tqdm.write((f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}'))
            # print("Samples so far: %s" % ((step + 1) * batch_size))
 
    # Print epoch number and loss value at the end of every epoch
    print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result()))
 
    # Save a checkpoint after every five epochs
    if (epoch + 1) % 5 == 0:
        save_path = ckpt_manager.save()
        print("Saved checkpoint at epoch %d" % (epoch + 1))
 
print("Total time taken: %.2fs" % (time() - start_time))

for (step, (train_batchX, train_batchY)) in enumerate(train_dataset):

  print(step, train_batchX.shape, train_batchY.shape)
  break