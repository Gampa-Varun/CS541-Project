
from pickle import load
from tensorflow import Module
import tensorflow as tf
#from keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64, TensorArray, argmax, newaxis, transpose
from transformer import TransformerModel
from keras.applications.vgg16 import preprocess_input
#from dataloader import load_image


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image,(384,384))
    image = tf.image.per_image_standardization(image)
    image = preprocess_input(image)
    #print("sdfdsf", image)
    return image, image_path

dec_vocab_size = 8918
dec_seq_length = 39


# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack
dropout_rate = 0.0

inferencing_model = TransformerModel(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)

class Translate(Module):
    def __init__(self, inferencing_model, **kwargs):
        super(Translate, self).__init__(**kwargs)
        self.transformer = inferencing_model
 
    def load_tokenizer(self, name):
        with open(name, 'rb') as handle:
            return load(handle)
    def padding_train_sequences(train_seqs, max_length, padding_type):
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding=padding_type,maxlen=max_length)
        return cap_vector
           
#padded_caption_vector = padding_train_sequences(train_seqs,max_length,'post')
 
    def __call__(self, sentence):
        # Append start and end of string tokens to the input sentence
        #sentence[0] = "<START> " + sentence[0] + " <EOS>"
 
        # Load encoder and decoder tokenizers
        dec_tokenizer = self.load_tokenizer('dec_tokenizer.pkl')
 

 
        # Prepare the output <START> token by tokenizing, and converting to tensor
        output_start = dec_tokenizer.texts_to_sequences(["<start>"])
        output_start = convert_to_tensor(output_start[0], dtype=tf.int32)
 
        # Prepare the output <EOS> token by tokenizing, and converting to tensor
        output_end = dec_tokenizer.texts_to_sequences(["<end>"])
        output_end = convert_to_tensor(output_end[0], dtype=tf.int32)
 
        # Prepare the output array of dynamic size
        decoder_output = TensorArray(dtype=int64, size=0, dynamic_size=True)
        decoder_input = tf.Variable(tf.zeros( (38),dtype=tf.int32 ), name="decoder_input")
        #decoder_output = decoder_output.write(0, output_start)
        decoder_input = decoder_input[0].assign(output_start)
        decoder_input_ = tf.expand_dims(decoder_input, axis=0)
        for i in range(dec_seq_length-1):
 

            prediction = self.transformer(sentence, decoder_input_, training=False)
            print("prediction shape: ", prediction.shape)
             

 
            prediction = prediction[:, i, :]
 
            # Select the prediction with the highest score
            predicted_id = argmax(prediction, axis=-1)
            #predicted_id = predicted_id[0][newaxis]
 
            # Write the selected prediction to the output array at the next available index
            #decoder_input = tf.squeeze(decoder_input)
            #decoder_output = decoder_output.write(i + 1, predicted_id)
            predicted_id = tf.cast(predicted_id, tf.int32)
            decoder_input = decoder_input[i].assign(predicted_id)
            decoder_input_ = tf.expand_dims(decoder_input, axis=0)
 
            # Break if an <EOS> token is predicted
            if predicted_id == output_end:
                break
 
        #output = transpose(decoder_output.stack())[0]
        output = decoder_input.numpy()
 
        output_str = []
 
        # Decode the predicted tokens into an output string
        for i in range(len(output)):
 
            key = output[i]
            #print(dec_tokenizer.index_word[key])
            output_str.append(dec_tokenizer.index_word[key])
            if(key == output_end):
                break
 
        return output_str



# Load the trained model's weights at the specified epoch
#inputs = tf.keras.Input(shape=(3,384,384), name="original_img")
#inputs2 = tf.keras.Input(shape=(38),name='text')
#outputs = inferencing_model(inputs,inputs2)
#inferencing_model = tf.keras.Model([inputs,inputs2],outputs,name ='inferencing_model')
#inferencing_model = tf.keras.models.load_model('saved_model/my_model')
image_path = 'Dataset/Flicker8k_Dataset/3637013_c675de7705.jpg'

image, image_path = load_image(image_path) 
# Create a new instance of the 'Translate' class
image = tf.expand_dims(image,axis=0)
image = tf.cast(image,tf.float32)
image = tf.transpose(image,perm=[0,3,1,2])
translator = Translate(inferencing_model)
 
# Translate the input sentence

print(translator(image))