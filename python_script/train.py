import os
from KMeanTraining import KMeanTextClustering

######## Unsupervised Learning ########
dataset_path = os.getcwd().split(os.sep)[:-1] + ['dataset', 'training_dir', 'dummy.json']
dataset_path = os.sep.join(dataset_path)
saved_model_path = os.getcwd().split(os.sep)[:-1] + ['cluster_model']
saved_model_path = os.sep.join(saved_model_path)
tokenized_column = ['description']
model = KMeanTextClustering(config_dict={},train_file_path=dataset_path,saved_model_path=saved_model_path,tokenized_column=tokenized_column)
model.word_cut()
model.train_and_export_model()
######################################