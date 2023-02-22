from pickle import load
from matplotlib.pylab import plt
from numpy import arange
import numpy as np
 
# Load the training and validation loss dictionaries
train_loss = load(open('train_loss.pkl', 'rb'))
val_loss = load(open('val_loss.pkl', 'rb'))

train_loss_prev = load(open('train_loss1.pkl','rb'))
val_loss_prev = load(open('val_loss1.pkl', 'rb'))
 
# Retrieve each dictionary's values
train_values = train_loss.values()
val_values = val_loss.values()

train_values_prev = train_loss_prev.values()
val_values_prev  = val_loss_prev.values()

train_values_mid = np.asarray([3.2,3.18,3.13,3.07,3.02,2.98,2.95,2.92,2.9,2.88,2.88])
val_values_mid = np.asarray([3.35,3.3,3.25,3.2,3.18,3.13,3.11,3.07,3.05,3.01,2.95])


 
# Generate a sequence of integers to represent the epoch numbers
#epochs = range(1, 20)

train = []
val = []
epochs = range(1,41)

for i in train_values_prev:
	train.append(i.numpy())

for i in train_values_mid:
	train.append(i)

for i in train_values:
	train.append(i.numpy())



for i in val_values_prev:
	val.append(i.numpy())

for i in val_values_mid:
	val.append(i)

for i in val_values:
	val.append(i.numpy())


train_values = train
val_values = val
 
# Plot and label the training and validation loss values
plt.plot(epochs, train_values, label='Training Loss')
plt.plot(epochs, val_values, label='Validation Loss')
 
# Add in a title and axes labels
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
 
# Set the tick locations
plt.xticks(arange(0, 41, 2))
 
# Display the plot
plt.legend(loc='best')
plt.show()