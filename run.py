# coding: UTF-8
import torch
import numpy as np
from train_eval import train, init_network,predict
from utils import My_Dataset,build_vocab
from TextCNN import *
from torch.utils.data import DataLoader
import pandas as pd

def run(model_name, file):
    dataset = 'data'  # 数据集
    config = Config(dataset, model_name=model_name, file=file)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    print("Loading data...")

    vocab = build_vocab(config)
    train_data = My_Dataset(config, config.train_path, vocab)
    dev_data = My_Dataset(config, config.dev_path, vocab)
    test_data = My_Dataset(config, config.test_path, vocab)
    # predict_data = My_Dataset(config, config.predict_path, vocab)

    train_iter = DataLoader(train_data, batch_size=config.batch_size)
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size)
    test_iter = DataLoader(test_data, batch_size=config.batch_size)
    # predict_iter = DataLoader(predict_data, batch_size=config.batch_size)
    # train
    config.n_vocab = len(vocab)
    TextCNN_model = Model(config)
    ## 模型放入到GPU中去
    TextCNN_model = TextCNN_model.to(config.device)
    print(TextCNN_model.parameters)
    train(config, TextCNN_model, train_iter, dev_iter, test_iter)

if __name__ == '__main__':
    run(model_name='prefer', file='prefer')





