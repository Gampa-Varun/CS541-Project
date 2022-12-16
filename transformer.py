from decoder import Decoder
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from SWINblock import SwinTransformer
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
 
 
class TransformerModel(Model):
    def __init__(self,  dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate, name=None,**kwargs):
        super(TransformerModel, self).__init__(name=name,**kwargs)
 
        # Set up the encoder
        self.encoder = SwinTransformer(name=name+'SWINblock')
 
        # Set up the decoder
        self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate,name = name+'Decoder')
 
        # Define the final dense layer
        self.model_last_layer = Dense(dec_vocab_size,name=name+'Dense')

        self.dec_seq_length = dec_seq_length
 
    def padding_mask(self, input):
        # Create mask which marks the zero padding values in the input by a 1.0
        mask = math.equal(input, 0)
        mask = cast(mask, float32)
 
        # The shape of the mask should be broadcastable to the shape
        # of the attention weights that it will be masking later on
        return mask[:, newaxis, newaxis, :]
 
    def lookahead_mask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)
 
        return mask


    def build_graph(self,training):
        input_layer1 = tf.keras.Input(shape=(self.dec_seq_length-1))
        input_layer2 = tf.keras.Input(shape=(3,224,224))
        return Model(inputs=[input_layer2,input_layer1], outputs=self.call([input_layer2, input_layer1],training))

    def build(self, layer):
        if isinstance(layer, tf.keras.layers.Dense):
            layer.kernel_initializer = TruncatedNormal(stddev=0.02,name=layer.name+'dense')
            #print(layer.name)
            if layer.bias is not None:
                layer.bias_initializer = Constant(0,name=layer.name+'bias')
        elif isinstance(layer, tf.keras.layers.LayerNormalization):
            layer.bias_initializer = Constant(0,name=layer.name+'beta')
            layer.gamma_initializer = Constant(1.0,name=layer.name+'gamma')

    def tensorplot(self,inputtensor,channel=0,name=None):
        check = inputtensor.numpy()

        B,L,C = np.shape(check)
        H = np.sqrt(L)
        H = H.astype(int)
        W = H
        check = np.reshape(check,[1,H,W,C])



        check1 = check[0]

        check1 = np.transpose(check1, axes=[2,0,1])

        check1 = check1[channel]
        mincheck1 = np.min(check1)

        check1 = check1- mincheck1

        maxcheck = np.max(check1)

        check1 = check1/maxcheck

        #print(check1)

        check1 = np.expand_dims(check1,axis=0)

        check1 = np.transpose(check1,axes=[1,2,0])


        fig, ax1 = plt.subplots(1,1)

        im = ax1.imshow(check1, interpolation='nearest')

        ax1.set_title(name, color='black')



        plt.imshow(check1,cmap='gray')
        plt.show()


 
    def call(self, inputs, training):

        encoder_input, decoder_input = inputs
 
        # Create padding mask to mask the encoder inputs and the encoder outputs in the decoder

 
        # Create and combine padding and look-ahead masks to be fed into the decoder
        dec_in_padding_mask = self.padding_mask(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)
 
        # Feed the input into the encoder
        encoder_output, encoder_output_global = self.encoder(encoder_input,training)

        #self.tensorplot(encoder_output,channel=0,name='Encoder output (post refining layer)')
        #print("enc output: ", encoder_output)
        #print(" glob output: ", encoder_output_global)
 


        # Feed the encoder output into traininghe decoder
        decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, encoder_output_global, training)
 
        # Pass the decoder output through a final dense layer
        model_output = self.model_last_layer(decoder_output)
 
        return model_output
