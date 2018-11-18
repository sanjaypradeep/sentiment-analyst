import pandas as pd
import os
from sentimentanalyser import preprocess
# from scripts.preprocess import PreProcess


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

# from . import constants
from copy import copy,deepcopy

class TestData():

    def pre_process_data(self,data):

        #the pre processing part
        column_name = data.columns[0]
        # data=df
        pre_processor = preprocess.PreProcess(data, column_name)    

        data = pre_processor.clean_html()
        data = pre_processor.remove_non_ascii()
        data = pre_processor.remove_spaces()
        data = pre_processor.remove_punctuation()
        data = pre_processor.stemming()
        data = pre_processor.lemmatization()
        data = pre_processor.stop_words()

        return data



    def test_model(self,test_text, test_file_name, test_reference_file,outputDir):
        #returns dataframe with label, status as True or False, 

        print("text is ",test_text," test file is ",test_file_name," reference file is ",test_reference_file)
        # print("constant = ",constants.vectorlibs_location,constants.trained_models_location)
        identifier=test_reference_file



        ###############################################################################
        # Set up source areas/output areas
        ###############################################################################

        # folder,fileNameEx=os.path.split(test_reference_file)
        # referFilenameNoExtn=fileNameEx.split(".")[0]
        referFilenameNoExtn=test_reference_file
        print("Reference filename without extn is ",referFilenameNoExtn)


        storage_location=str(outputDir)+"/"+referFilenameNoExtn+"/"
        if not os.path.exists(storage_location):
            print("cannot find pkls at ",storage_location)


        if test_file_name!=None:
            folderTest,fileNameExTest=os.path.split(test_file_name)
            testFilenameNoExtn=fileNameExTest.split(".")[0]
        else:
            testFilenameNoExtn="None"




        





        test_is_a_file=False
        #this flag becomes true if test data is a csv file
        #also enabbles the result dataframe to be written to 
        #another csv file
        #you give csv, you get csv

        if test_file_name is None and test_text!= "":            
            print("Getting text",len(test_text))
            # text entered on the Text Box
            if len(test_text) < 20:
                print("Please provide input large enough, Classifier can understand :)")
                return None,None,False
            else:
                print("Generating the dataframe from text")
                data = {'Text': [test_text]}
                dataFrame = pd.DataFrame(data=data)
                print("Done converting string to dataframe")
                print(dataFrame.head())

        #need to consider file
        elif test_file_name != None and  test_text=="":
            print("test file name is ",test_file_name)
            test_is_a_file=True
            dataFrame=pd.read_csv(test_file_name)
            print(dataFrame.head(5))


        else:
            print("What am i doing here")
            return None,None,False


        #keep a backup of df

        dataFramebackUp=deepcopy(dataFrame)



        #now work on the dataframe df
        vectorizer= storage_location+'vectorizer.pkl'
        print("the vectorizer is "+str(vectorizer))
        tfidf_transformer = joblib.load(vectorizer)
        
        print("Loaded vectorizer ",vectorizer)
        print(vectorizer)

        #pre process data 
        data=self.pre_process_data(dataFrame)
        print("After pre processing")
        print(data.head(5))

        column1=data.columns[0]
        data_check = tfidf_transformer.transform(data[column1])
        print("The data after pre process and transform is")
        print(data.head(5))

        print("data check type is :",type(data_check))
        print("Backup looks like this")
        print(dataFramebackUp.head(5))

        for model_name in ['SVM','Naive-Bayes']:
            model_file= storage_location+str(model_name)+".pkl"
            print("going for model pkl at",model_file)
            model=joblib.load(model_file)
            print("Loaded ",model_file)
            output=model.predict(data_check)
            print("After running model")
            print(output)
            dataFramebackUp[model_name]=output

        print("After testing result is")
        print(dataFramebackUp.head())

        #write to file
        dataFramebackUp.to_csv(storage_location+ str(testFilenameNoExtn)+"_results.csv")
        print("result stored at "+storage_location+ str(testFilenameNoExtn)+"_results.csv")
        print("_______________________")
        return dataFramebackUp
    


# if __name__ == "__main__":


#     objTest = TestData()

#     testText=""
#     test_file_name="/Users/amirulislam/projects/built_apps/doc_classific_expanded/source samples/twitter_test_unlabeled.csv"
#     test_reference_file="/Users/amirulislam/projects/built_apps/doc_classific_expanded/source samples/twitter_train.csv"
#     outputDir="/Users/amirulislam/Desktop/outputs"

#     testedDataFrame=objTest.test_model(testText,test_file_name,test_reference_file,outputDir)


#     print("result for input file is")
#     print(testedDataFrame.head())

#     testText = '''
#     Sir Kenny received the honour from Prince Charles during a ceremony at Buckingham Palace.

# As a player, Sir Kenny helped Liverpool win three European Cups in seven years. He went on to successfully manage both Liverpool and Blackburn Rovers.

# Sir Kenny supported the families of the victims of the Hillsborough disaster and he and his wife Marina have raised more than £10m for charity.

# Also being honoured during the investiture at Buckingham Palace, were actor Tom Hardy, footballer Jermain Defoe, television historian Lucy Worsley and entrepreneur Jo Malone.

# Kenny Dalglish's honour was for "services to football, charity, and the city of Liverpool".
# '''
#     test_file_name = None
#     test_reference_file = "/Users/amirulislam/projects/built_apps/doc_classific_expanded/source samples/bbc_dataset.csv"
#     outputDir="/Users/amirulislam/Desktop/outputs"
    
#     testedDataFrame = objTest.test_model(testText, test_file_name, test_reference_file,outputDir)