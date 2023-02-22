
from pickle import load
from tensorflow import Module
import tensorflow as tf
#from keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64, TensorArray, argmax, newaxis, transpose
from transformer import TransformerModel
from keras.applications.vgg16 import preprocess_input
#from dataloader import load_image
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import Adam
from tensorflow import cast
from datasetpath import image_path, text_path
from dataloader import ddd
import numpy as np

from operator import itemgetter

import sys, time, os, warnings 
warnings.filterwarnings("ignore")

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image,(224,224))
    #image = tf.image.per_image_standardization(image)
    #image = preprocess_input(image)
    image = tf.divide(image,255.0)


    print(image)

    return image, image_path

def accuracy_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of accuracy
    padding_mask = math.logical_not(equal(target, 0))
 
    # Find equal prediction and target values, and apply the padding mask
    maxpred = cast(argmax(prediction, axis=2),tf.int32)
    accuracy = equal(target, maxpred)
    accuracy = math.logical_and(padding_mask, accuracy)
 
    # Cast the True/False values to 32-bit-precision floating-point numbers
    padding_mask = cast(padding_mask, tf.float32)
    accuracy = cast(accuracy, tf.float32)
 
    # Compute the mean accuracy over the unmasked values
    return reduce_sum(accuracy) / reduce_sum(padding_mask)



def loss_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of loss
    padding_mask = math.logical_not(equal(target, 0))
    padding_mask = cast(padding_mask, tf.float32)
 
    # Compute a sparse categorical cross-entropy loss on the unmasked values
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask
 
    # Compute the mean loss over the unmasked values
    return reduce_sum(loss) / reduce_sum(padding_mask)

beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-5
dropout_rate = 0.1
d_model = 512

class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)
 
        self.d_model = cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
 
    def __call__(self, step_num):
 
        # Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)
 
        return (self.d_model ** -0.5) * math.minimum(arg1, arg2)

# Instantiate an Adam optimizer
optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)



dec_vocab_size = 8000
dec_seq_length = 39


# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack
dropout_rate = 0.0

inferencing_model = TransformerModel(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate,name='SWINtransformer').build_graph(False)
inferencing_model.compile(loss=loss_fcn, optimizer=optimizer)


#checkpoint = tf.train.Checkpoint()

# Use the Checkpoint.restore method to restore the model weights from the checkpoint file
#checkpoint.restore(tf.train.latest_checkpoint('weights'))


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)

    pattern = r"[^A-Za-z0-9_<>]"

    stripped_lowercase = tf.strings.regex_replace(lowercase, pattern," ")

    return stripped_lowercase

class Translate(Module):
    def __init__(self, inferencing_model, **kwargs):
        super(Translate, self).__init__(**kwargs)
        self.transformer = inferencing_model
 
    def load_tokenizer(self, name):
        images_dir = image_path

        captions_dir = text_path
        batch_size = 2
        buffer_size = 100
        dec_token_weights_config = load(open("dec_tokenizer.pkl", "rb"))
        print(images_dir)
        train_dataset, val_dataset, test_dataset, vectorizer = ddd(images_dir, captions_dir, batch_size, buffer_size,dec_token_weights_config)

        return vectorizer


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
        output_start = dec_tokenizer("<start>")
        output_start = cast(output_start, dtype=tf.int32)

        # output_start_2 = dec_tokenizer("<start>")
        # output_start_2 = cast(output_start, dtype=tf.int32)
 
        # Prepare the output <EOS> token by tokenizing, and converting to tensor
        output_end = dec_tokenizer("<end>")
        output_end = cast(output_end, dtype=tf.int32)
 
        # Prepare the output array of dynamic size
        decoder_output = TensorArray(dtype=int64, size=0, dynamic_size=True)
        decoder_input = tf.Variable(tf.zeros( (38),dtype=tf.int32 ), name="decoder_input")

        # decoder_input_2 = tf.Variable(tf.zeros( (38),dtype=tf.int32 ), name="decoder_input")
        #decoder_output = decoder_output.write(0, output_start)
        decoder_input = decoder_input[0].assign(output_start)


        decoder_input_ = tf.Variable(tf.expand_dims(decoder_input, axis=0))

        decoder_input_list = [decoder_input_ for i in range(3)]
        score = [ 0.0 for i in range(3)]

        end_reached = []

        num_reached = 0
        

        for i in range(37):

            prediction = []

            for decoder_input_ in decoder_input_list:
                prediction.append(tf.nn.softmax(self.transformer([sentence, decoder_input_], training=False)))

            #decoder_input_list=[]

            # print("prediction shape: ", prediction.shape)
            topk = [tf.argsort(prediction[j][0,i,:]) for j in range(len(decoder_input_list))]
            # topk1 = tf.argsort(prediction[0])
            # topk2 = tf.argsort(prediction[1])
            # topk3 = tf.argsort(prediction[2])
            #print(topk[0][-3:-1])

            candidate = [itemgetter(int(topk[j][-3]),int(topk[j][-2]),int(topk[j][-1]))(prediction[j][0][i]) for j in range(len(decoder_input_list))] # int(topk[j][-3]),int(topk[j][-2]),int(topk[j][-1])

            # candidate1 = prediction[0](topk1[-3:-1])
            # candidate2 = prediction[1](topk2[-3:-1])
            # candidate3 = prediction[2](topk3[-3:-1])

            maxval1 = -float('inf')
            maxval2 = -float('inf')
            maxval3 = -float('inf')
            # print(candidate)

            k_indices = [[0,0],[0,0],[0,0]]


            for j in range(len(decoder_input_list)):
                lengthj = sum(bool(x) for x in decoder_input_list[j][0])
                print(lengthj)
                lengthj_1 = 1.0
                if lengthj == 1:
                    lengthj_1 = 1
                else:
                    lengthj_1 = lengthj - 1

                lengthj = np.power(lengthj,0.0)
                lengthj_1 = np.power(lengthj_1, 0.0)
                for k in range(3):
                    if (score[j]+tf.math.log(candidate[j][k]))*lengthj_1/lengthj > maxval1 and (score[j]+ tf.math.log(candidate[j][k]))*lengthj_1/lengthj > maxval2 and (score[j]+ tf.math.log(candidate[j][k]))*lengthj_1/lengthj > maxval3:
                        maxval1 = (score[j] + tf.math.log(candidate[j][k]))*lengthj_1/lengthj
                        k_indices[0] = [j,k]

                    elif (score[j]+ tf.math.log(candidate[j][k]))*lengthj_1/lengthj > maxval2 and (score[j]+ tf.math.log(candidate[j][k]))*lengthj_1/lengthj > maxval3 and (score[j]+ tf.math.log(candidate[j][k]))*lengthj_1/lengthj < maxval1:
                        maxval2 = (score[j] + tf.math.log(candidate[j][k]))*lengthj_1/lengthj
                        k_indices[1] = [j,k]

                    elif (score[j]+ tf.math.log(candidate[j][k]))*lengthj_1/lengthj > maxval3 and (score[j]+ tf.math.log(candidate[j][k]))*lengthj_1/lengthj < maxval2 and (score[j]+ tf.math.log(candidate[j][k]))*lengthj_1/lengthj < maxval1:
                        maxval3 = (score[j] + tf.math.log(candidate[j][k]))*lengthj_1/lengthj
                        k_indices[2] = [j,k]

            # for j in range(len(decoder_input_list)):
            #     for k in range(3):
            #         if score[j]*candidate[j][k] > maxval1 and score[j]*candidate[j][k] > maxval2 and score[j]*candidate[j][k] > maxval3:
            #             maxval1 = score[j]*candidate[j][k]
            #             k_indices[0] = [j,k]

            #         elif score[j]*candidate[j][k] > maxval2 and score[j]*candidate[j][k] > maxval3 and score[j]*candidate[j][k] < maxval1:
            #             maxval2 = score[j]*candidate[j][k]
            #             k_indices[1] = [j,k]

            #         elif score[j]*candidate[j][k] > maxval3 and score[j]*candidate[j][k] < maxval2 and score[j]*candidate[j][k] < maxval1:
            #             maxval3 = score[j]*candidate[j][k]
            #             k_indices[2] = [j,k]


            score[0] = maxval1
            score[1] = maxval2
            score[2] = maxval3

            print(score)
                


            predicted_id1 = topk[k_indices[0][0]][-3+k_indices[0][1]]
            predicted_id2 = topk[k_indices[1][0]][-3+k_indices[1][1]]
            predicted_id3 = topk[k_indices[2][0]][-3+k_indices[2][1]]


            predicted_id = []

            predicted_id.append(tf.cast(predicted_id1, tf.int32))
            predicted_id.append(tf.cast(predicted_id2, tf.int32))
            predicted_id.append(tf.cast(predicted_id3, tf.int32))

            to_remove = []

            for j in range(len(decoder_input_list)):

                decoder_input = tf.Variable(tf.squeeze(decoder_input_list[j]))
                decoder_input = decoder_input[i+1].assign(predicted_id[j])

   

                decoder_input_list[j] = tf.expand_dims(decoder_input, axis=0)

    
            # Break if an <EOS> token is predicted
                if predicted_id[j] == output_end:
                    end_reached.append([decoder_input, score[j]])
                    to_remove.append(j)


            for j in sorted(to_remove, reverse=True):
                for index in reversed(range(j,len(decoder_input_list))):
                    decoder_input_list.pop(index)
                    num_reached+=1
       
                

            if num_reached == 3:
                break


        index = 0
        maxscore = 0

        for i in range(len(end_reached)):
            if end_reached[i][1] > maxscore:
                index = i

        output = end_reached[i][0].numpy()
 
        output_str = []
 
        # Decode the predicted tokens into an output string
        for i in range(len(output)):
 
            key = output[i]
            #print(dec_tokenizer.index_word[key])
            output_str.append(dec_tokenizer.get_vocabulary()[key])
            if(key == output_end):
                break
 
        return output_str




#print(translator(image))

inferencing_model.load_weights('./checkpoints/my_checkpoint')

#print(translator(image))


image_file = 'Dataset/Flicker8k_Dataset/2867699650_e6ddb540de.jpg'

image, image_file = load_image(image_file) 
# Create a new instance of the 'Translate' class
image = tf.expand_dims(image,axis=0)
image = tf.transpose(image,perm=[0,3,1,2])
translator = Translate(inferencing_model)

print(translator(image))