# coding: UTF-8
import torch
import numpy as np
from train_eval import train, init_network,predict
from utils import My_Dataset,build_vocab
from TextCNN import *
from torch.utils.data import DataLoader

def predict(config, model, data_iter):
	print('开始读取模型',config.save_path)
	model.load_state_dict(torch.load(config.save_path))
	model.eval()
	predict_all = np.array([], dtype=int)
	with torch.no_grad():
		for texts, labels in data_iter:
			outputs = model(texts)
			predic = torch.max(outputs.data, 1)[1].cpu().numpy()
			predict_all = np.append(predict_all, predic)
	return predict_all

def pre(model_name,file):
	dataset = 'data'  # 数据集
	config = Config(dataset, model_name,file)
	np.random.seed(1)
	torch.manual_seed(1)
	torch.cuda.manual_seed_all(1)
	torch.backends.cudnn.deterministic = True  # 保证每次结果一样
	print("Loading data...")
	vocab = build_vocab(config)
	predict_data = My_Dataset(config, config.predict_path, vocab)
	predict_iter = DataLoader(predict_data, batch_size=config.batch_size)
	# predict
	config.n_vocab = len(vocab)
	TextCNN_model = Model(config)
	## 模型放入到GPU中去
	TextCNN_model = TextCNN_model.to(config.device)
	reasult = predict(config, model=TextCNN_model, data_iter=predict_iter)
	print(reasult)
	reasult = reasult.tolist()
	for i in reasult:

		with open(f'data\\datasets\\outputs\\{file}.txt', 'a') as f:
			f.write(str(i) + '\n')
	f.close()
if __name__ == '__main__':
	# pre(model_name= 'textcnn_i',file='i')
	pre(model_name= 'prefer',file='prefer')
	# pre(model_name= 'chen',file='chen')

	import pandas as pd
	word=[]
	prefer=[]
	with open(r'data\\datasets\\outputs\\prefer.txt','r')as f:
		lines=f.readlines()
		prefer_pre=[line.strip('\n') for line in lines]
	f.close()
	with open(r'data\\datasets\\prepare\\prefer\\predict.txt','r',encoding='utf-8')as f:
		lines = f.readlines()
		for line in lines:
			word.append(line.split('\t')[0])
			prefer.append(line.split('\t')[1].strip('\n'))
	f.close()
	dict_word = {'word': word, 'prefer': prefer, 'prefer_pre': prefer_pre}
	data = pd.DataFrame(dict_word)
	data.to_csv(r'data\data_pre.csv', index=False)







