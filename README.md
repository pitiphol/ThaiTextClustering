# Thai Text Clustering
kmean clustering thai text description.

## Usage
1. ใส่ dataset ของคุณในรูปแบบ ของ json file ที่ dataset/training_dir/ (ดู format ของ json ที่ไฟล์ dummy.json)
2. โปรแกรมจะทำการตัดคำและแบ่ง cluster ที่ column description
3. run file train.py โดยเปลี่ยนชื่อของ dataset file เป็นชื่อของ file ที่ต้องการ
4. ดูผลลัพธ์คำที่เป็น top term ของแต่ละ cluster ได้ที่ cluster_model/term_per_clusters.txt
```
ตัวอย่างไฟล์ term_per_clusters.txt
0,['หนังสือ', 'เล่ม', 'บ้าน', 'ไทย', 'งาน', 'สร้าง', 'ทำ', 'เรื่อง', 'เทคนิค', 'รูปแบบ']
1,['สอบ', 'ข้อสอบ', 'เตรียม', 'คู่มือ', 'สถาบัน', 'วิชา', 'แนว', 'center', 'ความรู้', 'best']
2,['ชีวิต', 'คน', 'หนังสือ', 'เล่ม', 'ตัวเอง', 'ความสุข', 'ความสำเร็จ', 'ทำ', 'ที่จะ', '่']
3,['รัก', 'คน', 'เรื่อง', 'ความรัก', 'เรื่องราว', 'ตัว', 'ร์', 'หัวใจ', '่', 'ไม่ได้']
4,['เมนู', 'อาหาร', 'อร่อย', 'สูตร', 'ทำ', 'เค้ก', 'ขนม', 'จาน', 'รสชาติ', 'เล่ม']
5,['the', 'and', 'to', 'a', 'of', 'in', 'is', 'with', 's', 'for']
6,['สุขภาพ', 'โรค', 'ร่างกาย', 'อาหาร', 'ดี', 'ดูแล', 'หนังสือ', 'อาการ', 'กิน', 'สมุนไพร']
7,['ปลูก', 'สวน', 'ต้นไม้', 'มะนาว', 'ไม้', 'บ้าน', 'พืช', 'ผัก', 'พื้นที่', 'ชนิด']
8,['เที่ยว', 'เดินทาง', 'เมือง', 'ท่องเที่ยว', 'สถานที่ท่องเที่ยว', 'ข้อมูล', 'ประเทศ', 'ญี่ปุ่น', 'เล่ม', 'สถานที่']
9,['ลูก', 'คุณแม่', 'เลี้ยงลูก', 'พ่อแม่', 'เด็ก', 'คุณพ่อ', 'แม่', 'หนังสือ', 'พัฒนาการ', 'ดี']
10,['โลก', 'ผี', 'ไดโนเสาร์', 'เรื่อง', 'เรื่องราว', 'มนุษย์', 'หนังสือ', 'ตำนาน', 'หลอน', 'วิทยาศาสตร์']
11,['ธุรกิจ', 'หุ้น', 'การลงทุน', 'ขาย', 'สินค้า', 'ออนไลน์', 'หนังสือ', 'เล่ม', 'ลงทุน', 'สร้าง']
12,['ครู', 'สอบ', 'ผู้ช่วย', 'สพฐ', 'ข้อสอบ', 'หนังสือ', 'ภาคก', 'สังกัด', 'หลักสูตร', 'เล่ม']
13,['ธรรม', 'พระ', 'ธรรมะ', 'ท่าน', 'ปู่', 'หลวง', 'จิต', 'คำสอน', 'พุทธ', 'หนังสือ']
14,['เด็ก', 'ฝึก', 'ทักษะ', 'ภาษา', 'เด็กๆ', 'ภาษาอังกฤษ', 'สนุก', 'คำศัพท์', 'อ่าน', 'การเรียนรู้']
15,['โปรแกรม', 'การใช้งาน', 'ใช้งาน', 'งาน', 'สร้าง', 'windows', 'การเขียน', 'เรียนรู้', 'ตัวอย่าง', 'พื้นฐาน']
16,['ผม', 'คน', 'เรื่อง', 'ทำ', 'ไม่ได้', 'หนังสือ', 'ชีวิต', 'เล่ม', '่', 'ตัวเอง']
17,['พระ', 'สมเด็จ', 'พระองค์', 'บาท', 'ราช', 'พระเจ้าอยู่หัว', 'รม', 'เดช', 'พลอด', 'ภูมิ']
18,['ราศี', 'ดวง', 'ดวงชะตา', 'พยากรณ์', 'เสริม', 'ชะตา', 'มงคล', 'คำพยากรณ์', 'ปีน', 'เคล็ด']
19,['ถัก', 'โครเชต์', 'ลาย', 'กระเป๋า', 'งาน', 'นิต', 'ชิ้นงาน', 'ตุ๊กตา', 'ติ', 'ไหมพรม']
```
5. ลองแบ่งประโยคโดย run file example.py
```python
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
######################################
```
output
```python
[4]
```

## Requirements
- python 3
- pip
	- pandas
	- pythainlp (หรือ library ตัดคำอื่นๆเช่น deepcut)
	- scikit-learn

## CR.
- Library ตัดประโยคภาษาไทย และ stopword (https://github.com/PyThaiNLP/pythainlp)
- บทความวิธีทำ text clustering (https://pythonprogramminglanguage.com/kmeans-text-clustering/)

