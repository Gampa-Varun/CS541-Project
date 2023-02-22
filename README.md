# CS541-Project


The CS541-Project repo contains the modules for an end-end transformer model for image caption generation. 
It used a SWIN transformer as it's backbone, followed by novel refining encoders and finally a decoder to generate image captions
It is based on the paper: 'End-to-End Transformer Based Model for Image Captioning' authored by Yiyu Wang, Jungang Xu, Yingfei Sun
Link: https://arxiv.org/abs/2203.15350

If you use this corpus / data:

Please cite: M. Hodosh, P. Young and J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artifical Intellegence Research, Volume 47, pages 853-899
http://www.jair.org/papers/paper3994.html


Captions, Dataset Splits, and Human Annotations :


Flickr8k.token.txt - the raw captions of the Flickr8k Dataset . The first column is the ID of the caption which is "image address # caption number"

Flickr8k.lemma.txt - the lemmatized version of the above captions 

Flickr_8k.trainImages.txt - The training images used in our experiments
Flickr_8k.devImages.txt - The development/validation images used in our experiments
Flickr_8k.testImages.txt - The test images used in our experiments

The engine.py trains the transformer model, the swin transformer encoder model is the SWINblock.py. The decoder is present in decoder.py. Transformer.py creates a model by combining the decoder and encoder. For testing and running the software, inference2.py can be used.

Example: 

![image](https://user-images.githubusercontent.com/36986358/220552966-dd28ab1d-01d5-4e3b-b6ea-be4795ed9683.png)

Result: ['<start>', 'a', 'white', 'dog', 'running', 'on', 'the', 'grass', '<end>']
  
  
To get the desired result, the proper model weights are required which are currently unavailable over this repository.
