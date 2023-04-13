# Sound Recognition AI
This project is a simple sound recognition AI system built using Keras and TensorFlow. It can recognize three different sounds: the hoot of an owl, a hunter's shot, and the chirping of a robin.

## Prerequisites
To run the program, you will need to have Python 3 and the following libraries installed:

- numpy
- tensorflow
- pandas
- librosa
- keras
## Getting Started
To start, you will need to have sound files of the three sounds you want to recognize. Place the sound files in the `data` directory under a subdirectory with the name of the sound. For example, if you want to recognize the hoot of an owl, create a subdirectory called `owl_hoot` in the `data` directory and place owl hoot sound files in it.

Run the `data_preprocessing.py` file to preprocess the sound files and create the necessary data files. The preprocessed data will be saved in the `data_preprocessed` directory.

Once the data preprocessing is complete, run the `model_training.py` file to train the AI system. The trained model will be saved in the `saved_models` directory.

Finally, run the `model_testing.py` file to test the accuracy of the trained AI system. You can also use the `prediction.py` file to make predictions on new sound files.