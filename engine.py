from SWINblock import *
from dataloader import *

if __name__ == '__main__':
    # Test to load some images alongside its 5 corresponding captions
    images_dir = 'Dataset/Flicker8k_Dataset'

    captions_dir = 'Dataset/Flickr8k_text/Flickr8k.token.txt'

    train_dataset, test_dataset = ddd(images_dir, captions_dir)

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

    # Instantiate an Adam optimizer
    optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)
    
    # Create model
    dec_vocab_size = 8918
    dec_seq_length = 39
    enc_vocab_size = 8918
    enc_seq_length = 39

    training_model = TransformerModel(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
    

    # Include metrics monitoring
    train_loss = Mean(name='train_loss')
    train_accuracy = Mean(name='train_accuracy')

    # Create a checkpoint object and manager to manage multiple checkpoints
    ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
    ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)

    start_time = time.time()

    for epoch in range(epochs):
 
        train_loss.reset_states()
        train_accuracy.reset_states()
    
        print("\nStart of epoch %d" % (epoch + 1))

        # Iterate over the dataset batches
        for (step, (train_batchX, train_batchY)) in tqdm(enumerate(train_dataset)):

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
    
            if step % 50 == 0:
                print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
                # print("Samples so far: %s" % ((step + 1) * batch_size))
    
        # Print epoch number and loss value at the end of every epoch
        print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result()))
    
        # Save a checkpoint after every five epochs
        if (epoch + 1) % 5 == 0:
            
            print("Saved checkpoint at epoch %d" % (epoch + 1))

        # Print epoch number and loss value at the end of every epoch
        print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result()))

        # Save a checkpoint after every five epochs
        if (epoch + 1) % 5 == 0:
            save_path = ckpt_manager.save()
            print("Saved checkpoint at epoch %d" % (epoch + 1))

    print("Total time taken: %.2fs" % (time() - start_time))