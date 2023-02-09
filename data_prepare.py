import pandas as pd
import os
import shutil
import difflib
import time
#填补缺失值
#文本相似度匹配填补缺失值
def similarity():
	s1=time.time()
	df = pd.read_csv(r'data\\clean.csv', encoding='utf-8')
	# df['label_j_new'] = df['label_j']
	list1 = df['new_text'][df['label_j'] == -1].tolist()
	list2 = df['new_text'][df['label_j'] == -1].index.tolist()

	def string_similar(s1, s2):
		return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

	similarity = 0
	tag = 0
	num=0
	for i in range(1):
		for j in range(len(df)):
			temp = string_similar(str(list1[i+501]), str(df['new_text'][j]))
			if temp > similarity and df['label_j'][j] != -1:
				similarity = temp
				tag = j
		print(list1[i+501])
		print(similarity)
		print(df['new_text'][tag])
		if similarity >= 0.1:
			print(df['label_j_new'][list2[i + 501]])
			df['label_j_new'][list2[i+501]] = df['label_j'][tag]
			print(df['label_j'][tag])
		similarity = 0
		tag = 0
		num=num+1
		if num%50==0:
			print(num,'!!!')
	# df.to_csv(r'data\\clean.csv',index=False)
	end=time.time()
	print(end-s1)

#关键词填补缺失值
def pick(keywords,num):
	df = pd.read_csv(r'data\\train_clean.csv', encoding='utf-8')
	list1 = df['new_text'][df['label_j'] == -1].tolist()
	list2 = df['new_text'][df['label_j'] == -1].index.tolist()
	ls = []
	for i in range(len(list1)):
		if any(keyword in list1[i] for keyword in keywords):
			# print(list1[i],df['label_j_new'][list2[i]])
			ls.append(list1[i])
			df['label_j_new'][list2[i]] = num
	# print(list1[i], df['label_j_new'][list2[i]])
	print(keywords,len(ls))
	df.to_csv(r'data\\train_clean.csv', index=False)
	return len(ls)

def missing():
	num=0
	num=num+pick(['乳腺囊肿','左乳囊肿'],0)
	num = num + pick(['纤维瘤', '乳腺结节', '乳腺增生'], 1)
	num=num+pick(['乳腺瘤','乳腺肿瘤'],3)
	num=num+pick(['产前检查','产前诊断'],4)
	num=num+pick(['先兆流产','流产'],6)
	num=num+pick(['剖腹产'],8)
	num=num+pick(['发育迟缓','生长缓慢'],9)
	num=num+pick(['孩子肺炎','儿童肺炎'],25)
	num=num+pick(['扁桃体炎'], 28)
	num=num+pick(['早孕反应'], 29)
	num=num+pick(['月经失调','经期紊乱','月经紊乱'], 30)
	num=num+pick(['桥本甲状腺炎','甲状腺炎'], 31)
	num=num+pick(['肠胃病','消化不良','幽门螺杆菌'], 32)
	num=num+pick(['甲减'], 35)
	num=num+pick(['甲状腺瘤','甲状腺肿瘤'], 38)
	num=num+pick(['甲状腺结节'], 39)
	num=num+pick(['痔疮'], 40)
	num=num+pick(['皮肤痒'], 42)
	num=num+pick(['羊水异常'], 48)
	num=num+pick(['肺癌','肺结节','胸闷'], 49)
	num=num+pick(['胃病'],50)
	num=num+pick(['腹泻','拉肚子'],53)
	num=num+pick(['半月板'],55)
	num=num+pick(['半月板'],55)
	num=num+pick(['半月板'],55)
	print(num)


#分词 停用词清洗
# 去除停用词
import re
def removePunctuation(query):
	# 去除标点符号（只留字母、数字、中文)
	if query:
		rule = re.compile(u"[^a-zA-Z0-9\u4e00-\u9fa5]")
		query = rule.sub('', query)
	return query

def del_stopwords(file,test=False):
	df = pd.read_csv(r'data\\datasets\\'+file, encoding='utf-8')
	df['title']=df['title'].fillna('')
	df['diseaseName']=df['diseaseName'].fillna('')
	df['hopeHelp']=df['hopeHelp'].fillna('')
	df['conditionDesc']=df['conditionDesc'].fillna('')
	df['text'] = df['diseaseName'] + ' ' + df['conditionDesc'] + ' ' + df['title'] + ' ' + df['hopeHelp']
	df['new_text'] = df['text'].apply(lambda x: removePunctuation(x))
	if test:
		df.to_csv(r'data\\datasets\\test_clean.csv', index=False)
	else:
		df.to_csv(r'data\\datasets\\train_clean.csv',index=False)

#训练集测试集验证集准备
def textclasstxt(file,x,train_ratio=0.7,dev_ratio=0.1):
	if os.path.exists('data/datasets/prepare/textclass/'+x+''):
		shutil.rmtree('data/datasets/prepare/textclass/'+x)  # 删除prepare下的文件
	os.makedirs('data/datasets/prepare/textclass/'+x)

	df = pd.read_csv(r'data\\datasets\\'+file, encoding='utf-8')

	content=df['new_text'].tolist()
	label1=df[x].tolist()
	index1 = int(len(label1) * train_ratio)
	index2 = int(len(label1) * (train_ratio + dev_ratio))  # 拿到训练集下标
	train = label1[:index1]
	test = label1[index1:index2]  # 测试集文件集合
	dev = label1[index2:]  # 测试集文件集合

	file=['train','test','dev']
	dict={}
	dict[file[0]] = train
	dict[file[1]] = test
	dict[file[2]] = dev

	dict1 = {}
	dict1[file[0]] = content[:index1]
	dict1[file[1]] = content[index1:index2]
	dict1[file[2]] = content[index2:]
	dire=r'data\\datasets\\prepare\\textclass\\'+x

	for i in file:
		with open(dire + f'\\{i}.txt', 'a', encoding='utf-8') as f1:
			f1.seek(0)
			f1.truncate()
		f1.close()
		for j in range(0, len(dict1[i])):
			with open(dire + f'\\{i}.txt', 'a', encoding='utf-8') as f:
				f.write(str(dict1[i][j])+'\t'+str(dict[i][j]) + '\n')

if __name__ == '__main__':
	# del_stopwords(file='data_train.csv')
	# del_stopwords(file='data_test.csv',test=True)
	# # similarity()
	# missing()
	textclasstxt(file='train_clean1.csv',x='label_j_new')
	# textclasstxt(file='train_clean.csv',x='label_i')
