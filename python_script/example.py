# -*- coding: utf-8 -*-
import os
from KMeanTraining import KMeanTextClustering
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UnicodeWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
saved_model_path = os.getcwd().split(os.sep)[:-1] + ['cluster_model']
saved_model_path = os.sep.join(saved_model_path)
model = KMeanTextClustering(saved_model_path)
description = '''หนังสือเล่มนี้ได้รวบรวมสเต๊กหลากหลายประเภท อาหารฝรั่ง อาหารเอเชีย หลากหลายชนิด ตลอดจนความรู้พื้นฐานในการทำอาหาร'''
print(model.get_cluster(description))