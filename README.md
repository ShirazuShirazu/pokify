# pokify
Pokemon Name Converter (NLP)

# What does the program do?

This program is used to generate a pokemon name out of an inputted person's name. 
It uses Natural Language Processing to process the name, tokenize it and then make predictions


# How does it work?

It does this by downloading a training on a dataset of all 800+ official pokemon released, by breaking the names up by syllables and training on it with a Deep Neural Network. 
The Neural Network makes embeddings and uses a bidirectional LSTM to learn the  sequence in which these syllables appear in the pokemon names.
After training by syllables, the inputted person's name is also broken up and the Machine Learning model appends its predictions to the end of the inputted name. 

This program uses Recursive Neural Networks and NLP, which is the basis of advanced trend analysis software that social media uses.
