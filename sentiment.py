import pandas as pd
import difflib


def get_eventORbehavior():
	list_word = []
	list_bio = []
	list_bio2 = []
	with open(r'data/ner_predict.utf8',encoding='utf-8')as f:
		lines=f.readlines()
		word = ''
		bio = ''
		bio2 = ''
		for line in lines:
			if line == '\n':
				list_word.append(word)
				list_bio.append(bio)
				list_bio2.append(bio2)
				bio = ''
				bio2 = ''
				word = ''
				continue
			else:
				word += line.split(' ')[0]
				if line.split(' ')[1] != 'O':
					bio += line.split(' ')[1]
				if line.split(' ')[2] != 'O\n':
					bio2 += line.split(' ')[2]
		list_bio = [i.replace('\n', ' ') for i in list_bio]
		list_bio2 = [i.replace('\n', ' ') for i in list_bio2]
		eventORbehavior = []
		eventORbehavior1 = []

		event_keywords = ['EventReasult', 'EventAgent']
		behavior_keywords = ['BehaviorBehave', 'BehaviorAgent']

		for i in list_bio:
			sen = ''
			if any(keyword in i for keyword in event_keywords):
				sen += 'event'
			if any(keyword in i for keyword in behavior_keywords):
				sen += 'behavior'
			if i=='':
				sen='none'
			eventORbehavior.append(sen)
		for i in list_bio2:
			sen = ''
			if any(keyword in i for keyword in event_keywords):
				sen += 'event'
			if any(keyword in i for keyword in behavior_keywords):
				sen += 'behavior'
			if i == '':
				sen='none'
			eventORbehavior1.append(sen)
		print(len(eventORbehavior))

		print(len(list_bio))
		print(len(list_word))
		print(len(list_bio2))
		print(len(eventORbehavior1))

		dict_word = {'word': list_word, 'eventORbehavior': eventORbehavior,'eventORbehavior_pre': eventORbehavior1}
		data = pd.DataFrame(dict_word)
		data.to_csv(r'data\data_pre1.csv', index=False)


def string_similar(s1, s2):
	return difflib.SequenceMatcher(None, s1, s2).quick_ratio()
#从结果中将预测和原始进行完全比较（每一类）
def get_eb_sum():
	paths = ['train', 'test', 'dev']
	list_word = []
	list_bio = []
	for path in paths:
		with open(f'data/example_{path}.txt', encoding="utf-8") as f:
			lines = f.readlines()
			word = ''
			bio = ''
			for line in lines:
				if line == '\n':
					list_word.append(word)
					list_bio.append(bio)
					bio = ''
					word = ''
					continue
				else:
					word += line.split('\t')[0]
					if line.split('\t')[1] != 'O\n':
						bio += line.split('\t')[1]
	list_bio = [i.replace('\n', ' ') for i in list_bio]
	eventORbehavior = []

	event_keywords = ['EventReasult', 'EventAgent']
	behavior_keywords = ['BehaviorBehave', 'BehaviorAgent']

	for i in list_bio:
		sen = ''
		if any(keyword in i for keyword in event_keywords):
			sen += 'event'
		if any(keyword in i for keyword in behavior_keywords):
			sen += 'behavior'
		if i == '':
			sen = 'none'
		eventORbehavior.append(sen)

	print(len(list_word))

	dict_word = {'word': list_word, 'eventORbehavior': eventORbehavior}
	data = pd.DataFrame(dict_word)

	print(data['eventORbehavior'].value_counts())
	data['prefer']='null'
	data['emotion']='null'
	prefer = pd.read_csv(r'data\data_pre.csv', encoding='utf-8')
	for i in range(len(data)):
		for j in range(len(prefer)):
			if data['word'][i] == prefer['word'][j]:
				data['prefer'][i] = prefer['prefer'][j]
		continue
	print('11')

	for i in range(len(data)):
		for j in range(len(prefer)):
			if data['prefer'][i] == 'null':
				temp = string_similar((data['word'][i]), str(prefer['word'][j]))
				if temp > 0.9:
					data['prefer'][i] = prefer['prefer'][j]
	print(data['prefer'].value_counts())
	data.to_csv(r'data\data_eb.csv', index=False)



def get_alleventORbehavior():
	list_word = []
	list_bio = []
	list_bio2 = []
	with open(r'data/ner_predict.utf8',encoding='utf-8')as f:
		lines=f.readlines()
		word = ''
		bio = ''
		bio2 = ''
		for line in lines:
			if line == '\n':
				list_word.append(word)
				list_bio.append(bio)
				list_bio2.append(bio2)
				bio = ''
				bio2 = ''
				word = ''
				continue
			else:
				word += line.split(' ')[0]
				bio += line.split(' ')[1]
				bio2 += line.split(' ')[2].strip('\n')
		list_bio = [i.replace('\n', ' ') for i in list_bio]
		list_bio2 = [i.replace('\n', ' ') for i in list_bio2]
		df=pd.DataFrame({'content':list_word,'bio':list_bio,'bio_pre':list_bio2})
		df.to_csv(r'data\binary.csv', index=False)

def binary1():
	bin=pd.read_csv(r'data\binary.csv', encoding='utf-8')
	bin['prefer']='null'
	bin['prefer_pre']='null'
	prefer=pd.read_csv(r'data\data_pre.csv', encoding='utf-8')
	for i in range(len(bin)):
		for j in range(len(prefer)):

			if bin['content'][i]==prefer['word'][j]:
				bin['prefer'][i] = prefer['prefer'][j]
				bin['prefer_pre'][i] = prefer['prefer_pre'][j]
		continue
	print('11')

	for i in range(len(bin)):
		for j in range(len(prefer)):
			if bin['prefer'][i]=='null':
				temp = string_similar((bin['content'][i]), str(prefer['word'][j]))
				if temp > 0.9:
					bin['prefer'][i] = prefer['prefer'][j]
					bin['prefer_pre'][i] = prefer['prefer_pre'][j]
	print(bin['prefer'].value_counts())
	print(bin['prefer_pre'].value_counts())
	bin.to_csv(r'data\binary.csv', index=False)
def binary2():
	bin = pd.read_csv(r'data\binary.csv', encoding='utf-8')
	event_keywords = 'Event'
	behavior_keywords = 'Behavior'
	bin['emo'] = 'null'
	bin['emo_pre'] = 'null'
	for i in range(len(bin)):
		if event_keywords in bin['bio'][i] and bin['prefer'][i] == 0:
			bin['emo'][i] = '喜悦'
		if event_keywords in bin['bio'][i] and bin['prefer'][i] == 1:
			bin['emo'][i] = '悲伤'
		if behavior_keywords in bin['bio'][i] and bin['prefer'][i] == 0:
			bin['emo'][i] = '赞赏'
		if behavior_keywords in bin['bio'][i] and bin['prefer'][i] == 1:
			bin['emo'][i] = '指责'

		if event_keywords in bin['bio_pre'][i] and bin['prefer_pre'][i] == 0:
			bin['emo_pre'][i] = '喜悦'
		if event_keywords in bin['bio_pre'][i] and bin['prefer_pre'][i] == 1:
			bin['emo_pre'][i] = '悲伤'
		if behavior_keywords in bin['bio_pre'][i] and bin['prefer_pre'][i] == 0:
			bin['emo_pre'][i] = '赞赏'
		if behavior_keywords in bin['bio_pre'][i] and bin['prefer_pre'][i] == 1:
			bin['emo_pre'][i] = '指责'
	print(bin['emo'].value_counts())
	print(bin['emo_pre'].value_counts())
	bin.to_csv(r'data\binary1.csv', index=False)

def binary3():
	bin = pd.read_csv(r'data\binary1.csv', encoding='utf-8')
	event_keywords = 'Event'
	behavior_keywords = 'Behavior'
	bin['binary']='null'
	bin['binary_pre']='null'
	for i in range(len(bin)):
		if event_keywords in bin['bio'][i] and bin['emo'][i]=='喜悦':
			bin['binary'][i]='one'
		if event_keywords in bin['bio'][i] and bin['emo'][i]=='悲伤':
			bin['binary'][i]='two'
		if behavior_keywords in bin['bio'][i] and bin['emo'][i]=='赞赏':
			bin['binary'][i]='three'
		if behavior_keywords in bin['bio'][i] and bin['emo'][i]=='指责':
			bin['binary'][i]='four'

		if event_keywords in bin['bio_pre'][i] and bin['bio_pre'][i]==bin['bio'][i] and bin['emo_pre'][i]=='喜悦':
			bin['binary_pre'][i]='one'
		if event_keywords in bin['bio_pre'][i] and bin['bio_pre'][i]==bin['bio'][i] and bin['emo_pre'][i]=='悲伤':
			bin['binary_pre'][i]='two'
		if behavior_keywords in bin['bio_pre'][i] and bin['bio_pre'][i]==bin['bio'][i] and bin['emo_pre'][i]=='赞赏':
			bin['binary_pre'][i]='three'
		if behavior_keywords in bin['bio_pre'][i] and bin['bio_pre'][i]==bin['bio'][i] and bin['emo_pre'][i]=='指责':
			bin['binary_pre'][i]='four'

	if bin['binary'][i]=='null':
		bin=bin.drop(i)
	print('hhhhhh',bin['binary'].value_counts())
	bin.to_csv(r'data\binary1.csv', index=False)


if __name__ == '__main__':
	# get_eventORbehavior()
	# df=pd.read_csv(r'data\data_pre1.csv')
	# # print(df['eventORbehavior'].value_counts())
	# # for i in range(len(df)):
	# # 	if df['eventORbehavior'][i]=='none' or df['eventORbehavior'][i]=='eventbehavior':
	# # 		df=df.drop(i)
	# # print(df['eventORbehavior'].value_counts())
	# # print(df['eventORbehavior_pre'].value_counts())
	# df.to_csv(r'data\data_pre2.csv', index=False)
	# df=pd.read_csv(r'data\data_pre2.csv')
	# print(df['eventORbehavior'].value_counts())
	# get_alleventORbehavior()
	# binary1()
	# binary2()
	binary3()
	# bin = pd.read_csv(r'data\binary.csv', encoding='utf-8')
	# print(bin['binary'].value_counts())



#