from unittest import TestCase
from .train import Train

class TestTrain(TestCase):

    def test_training_data(self):
    	filePath = input('Enter the filePath of the training data')
    	outputDir= input('Enter the outputDir to generate trained models')
    	objTrain = Train()
		s = objTrain.train_file_model(filePath,outputDir)
        self.assertTrue(isinstance(s, basestring))