import pandas as pd

import preprocess
## from preprocess import PreProcess

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

# from . import constants

class Train():

	def pre_process_data(self, df):

		#the pre processing part
		column_name = df.columns[0]
		data=df
		pre_processor = preprocess.PreProcess(data, column_name)	

		data = pre_processor.clean_html()
		data = pre_processor.remove_non_ascii()
		data = pre_processor.remove_spaces()
		data = pre_processor.remove_punctuation()
		data = pre_processor.stemming()
		data = pre_processor.lemmatization()
		data = pre_processor.stop_words()

		return data


	def get_train_vectors(self, data, identifier,outputDir):
		col1=data.columns[0]
		col2=data.columns[1]

		# train_x, test_x, train_y, test_y = train_test_split(data[col1], data[col2], test_size=0.00)
		# print(train_x.shape, train_y.shape)
		# print(test_x.shape, test_y.shape)
		tfidf_transformer = TfidfVectorizer(min_df=1)
		train_vectors = tfidf_transformer.fit_transform(data[col1])

		# joblib.dump(tfidf_transformer, str(outputDir)+str(identifier)+'_vectorizer.pkl')
		joblib.dump(tfidf_transformer, str(identifier)+'_vectorizer.pkl')
		
		return train_vectors




	#this function is the entry point for training
	def train_file_model(self, filePath,outputDir):

		print("inside the Train File Model ....")

		# print("constant = ",constants.vectorlibs_location,constants.trained_models_location)
		
		print("In train file model, going to read file",filePath)	

		# ['SVM', 'Naive-Bayes']
		#this identifier will be used to save the pkl files
		# identifier=filePath

		#read the file content.
		dataFrame=pd.read_csv(filePath)
		print(dataFrame.head())

		#calling 
		data=self.pre_process_data(dataFrame)

		
		

		###############################################################################
		# Feature extraction
		###############################################################################
		
		train_vectors= self.get_train_vectors(data,filePath,outputDir)

		print("After vect")
		print(data.head())

		# why only 2 columns all the time? what if the file has more than 2 columns?
		col1=data.columns[0]
		col2=data.columns[1]



		 ###############################################################################
		# Perform classification with SVM, kernel=linear
		for each_model in ['SVM', 'Naive-Bayes']:
			print(each_model)
			if each_model == "SVM":
				model = svm.SVC(kernel='linear')
				model.fit(train_vectors, dataFrame[col2])
				

			elif each_model=="Naive-Bayes":
				model = MultinomialNB()
				print("going to naive baye")
				model.fit(train_vectors, dataFrame[col2])
			else:
				return False

			print("Saving training file")
			dataFrame.to_csv(str(filePath)+"_training_file.csv")
			print("Train file saved")

			print("going to store in ",str(filePath)+"_"+str(each_model)+'.pkl')	
			joblib.dump(model,str(filePath)+"_"+str(each_model)+'.pkl')


		return True



if __name__ == "__main__":


	objTrain = Train()

	filePath="/Users/amirulislam/projects/built_apps/doc_classific_expanded/source samples/bbc_dataset.csv"
	outputDir="/Users/amirulislam/Desktop"

	objTrain.train_file_model(filePath,outputDir)




