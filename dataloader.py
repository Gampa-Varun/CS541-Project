import string
from numpy import array
from pickle import load
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
images_dir = '/Users/rohin/Documents/CS541-Project-main/Dataset/Flicker8k_Dataset'
images = listdir(images_dir)

captions_dir = '/Users/rohin/Documents/CS541-Project-main/Dataset/Flickr8k_text/Flickr8k.token.txt'

print("The number of jpg flies in Flicker8k: {}".format(len(images)))

text = load_dataset(captions_dir)
print(text[:410])

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
print(len(data_frame))
print(data_frame[:10])

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
    #print("The number of unique filenames : {}".format(len(filenames_unique)))

    count_dict = Counter(data.filename.values)
    #print("Confirming that all the keys have count value as 5")
    #print(count_dict)

    print("The number of captions per image")
    count = Counter(Counter(data.filename.values).values())
    #print(count)
    return filenames_unique

unique_filenames = utility_counter(data)

#Plot the images along with the corresponding 5 captions for better data visualization
#Insert reference url here
def data_show(data):
    pic_count = 5
    pixels_count = 224
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
        caption = '<start>' + caption + ' <end>'
        final_captions.append(caption)

    return final_captions

final_captions = caption_preprocessor(data)

print(final_captions[:10])

#Resize the image to 224*224*3
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image,(224,224))
    image = preprocess_input(image)
    return image, image_path

image1, image1_path = load_image("/Users/rohin/Documents/CS541-Project-main/Dataset/Flicker8k_Dataset/3439243433_d5f3508612.jpg")
print("Shape after resize :", image1.shape)
plt.imshow(image1)

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

#print(image_dataset)

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

train_seqs , tokenizer = tokenize_caption(5000,train_captions)

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
  return cap_vector
           
padded_caption_vector = padding_train_sequences(train_seqs,max_length,'post')
print(padded_caption_vector.shape)

img_name_train, img_name_test, caption_train, caption_test = train_test_split(img_name_vector,padded_caption_vector,test_size=0.2,random_state=0)
print("Training Data : X = {0},Y = {1}".format(len(img_name_train), len(caption_train)))
print("Test Data : X = {0},Y = {1}".format(len(img_name_test), len(caption_test)))

#Convert the whole dataset into .npy format
batch_size = 1
buffer_size = 1000

def load_npy(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap

def create_dataset(img_name_train,caption_train):
  dataset = tf.data.Dataset.from_tensor_slices((img_name_train, caption_train))

  # Use map to load the numpy files in parallel
  dataset = dataset.map(lambda item1, item2: tf.numpy_function(load_npy, [item1, item2], [tf.float32, tf.int32]))

  # Shuffle and batch
  dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset


modelvgg = tf.keras.applications.VGG16(include_top=True,weights=None) # for observation on shapes


image_model = tf.keras.applications.VGG16(include_top=False,weights='imagenet')
new_input = image_model.input # Any arbitrary shapes with 3 channels
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

from tqdm import tqdm

for img, path in tqdm(image_dataset):

  batch_features = image_features_extract_model(img)
  batch_features = tf.reshape(batch_features,(batch_features.shape[0], -1, batch_features.shape[3]))
  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8")
    np.save(path_of_feature, bf.numpy())

np_img =np.load('/Users/rohin/Documents/CS541-Project-main/Dataset/Flicker8k_Dataset/3338291921_fe7ae0c8f8.jpg.npy')

print(np_img)
print("Shape : {}".format(np_img.shape))

# Creating train and test dataset
train_dataset = create_dataset(img_name_train,caption_train)
test_dataset = create_dataset(img_name_test,caption_test)

for (step, (train_batchX, train_batchY)) in enumerate(test_dataset):
  pass