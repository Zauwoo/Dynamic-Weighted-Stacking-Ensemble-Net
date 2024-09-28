import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
from paddlenlp.transformers import XLNetForSequenceClassification, XLNetTokenizer
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.transformers import LinearDecayWithWarmup

# 调用XLNet模型，同时因为本任务的情感分类是2类，设置num_classes = 2
model = XLNetForSequenceClassification.from_pretrained('xlnet-large-cased', num_classes=2)
tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')

# 定义要进行分类的类别
label_list = ['negative', 'positive']
label_map = {label: idx for idx, label in enumerate(label_list)}
print(label_map)

# 定义数据集类
class EmotionData(Dataset):
    def __init__(self, path):
        super(EmotionData, self).__init__()
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            header = next(f)  # 跳过第一行
            for line in f:
                label, text_a = line.strip().split('\t')
                self.data.append((text_a, int(label)))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# 定义数据加载和处理函数
def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    text_a = example[0]
    encoded_inputs = tokenizer(text=text_a, max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    if not is_test:
        label = np.array([example[1]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids

# 数据加载函数dataloader
def create_dataloader(dataset, mode='train', batch_size=1, batchify_fn=None, trans_fn=None):
    if trans_fn:
        dataset = [trans_fn(example) for example in dataset]
    shuffle = True if mode == 'train' else False
    batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    return paddle.io.DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)

# 加载训练集、验证集和测试集
train_ds = EmotionData('train.tsv')
dev_ds = EmotionData('dev.tsv')
test_ds = EmotionData('test.tsv')

batch_size = 128  # 批处理大小，可根据训练环境条件，适当修改此项
max_seq_length = 128  # 文本序列截断长度

# 将数据处理成模型可读入的数据格式
trans_func = lambda example: convert_example(example, tokenizer=tokenizer, max_seq_length=max_seq_length)

batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack()  # labels
): [data for data in fn(samples)]

# 训练集迭代器
train_data_loader = create_dataloader(train_ds, mode='train', batch_size=batch_size, batchify_fn=batchify_fn,
                                      trans_fn=trans_func)
# 验证集迭器
dev_data_loader = create_dataloader(dev_ds, mode='dev', batch_size=batch_size, batchify_fn=batchify_fn,
                                    trans_fn=trans_func)
# 测试集迭器
test_data_loader = create_dataloader(test_ds, mode='test', batch_size=batch_size, batchify_fn=batchify_fn,
                                     trans_fn=trans_func)

# 定义超参，loss，优化器等
# 定义训练过程中的最大学习率
learning_rate = 2e-5
# 训练轮次
epochs = 5
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.01

num_training_steps = len(train_data_loader) * epochs
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)

# AdamW优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ])

criterion = paddle.nn.loss.CrossEntropyLoss()  # 交叉熵损失函数

# 定义多类分类 Accuracy, Precision 和 Recall 类
class Metric():
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = np.zeros(2)
        self.fp = np.zeros(2)
        self.tn = np.zeros(2)
        self.fn = np.zeros(2)

    def update(self, preds, labels):
        preds = preds.argmax(axis=1).numpy()
        labels = labels.numpy()
        for i in range(2):
            self.tp[i] += ((preds == i) & (labels == i)).sum()
            self.fp[i] += ((preds == i) & (labels != i)).sum()
            self.tn[i] += ((preds != i) & (labels != i)).sum()
            self.fn[i] += ((preds != i) & (labels == i)).sum()

    def accumulate(self):
        precision = self.tp / (self.tp + self.fp + 1e-5)
        recall = self.tp / (self.tp + self.fn + 1e-5)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + 1e-5)
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        avg_accuracy = np.mean(accuracy)
        return avg_accuracy, avg_precision, avg_recall, avg_f1

metric = Metric()

# 定义模型训练验证评估函数
@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        metric.update(logits, labels)
    accu, precision, recall, f1 = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f, precision: %.5f, recall: %.5f, f1: %.5f" % (
    np.mean(losses), accu, precision, recall, f1))
    model.train()
    return accu, precision, recall, f1, np.mean(losses)  # 返回准确率、精度、召回率、F1值和损失

# 模型训练
save_dir = "checkpoint"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

pre_accu = 0
best_epoch = 0
best_precision = 0
best_f1 = 0
best_recall = 0
best_loss = float('inf')
global_step = 0
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)
        metric.update(probs, labels)
        acc, precision, recall, f1 = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0:
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, f1: %.5f" % (global_step, epoch, step, loss, acc, f1))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

    # 每轮结束对验证集进行评估
    accu, precision, recall, f1, loss = evaluate(model, criterion, metric, dev_data_loader)
    print(f'Epoch {epoch}, Accuracy: {accu:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Loss: {loss:.4f}')
    if accu > pre_accu:
        # 保存较上一轮效果更优的模型参数
        save_param_path = os.path.join(save_dir,
                                       f'XLNet_E{epoch}_accu_{accu:.4f}_F1_{f1:.4f}_loss_{loss:.4f}_recall_{recall:.4f}.pdparams')  # 保存模型参数
        paddle.save(model.state_dict(), save_param_path)
        pre_accu = accu
        best_epoch = epoch
        best_precision = precision
        best_f1 = f1
        best_recall = recall
        best_loss = loss

print(f'Best model from Epoch {best_epoch}, Accuracy: {pre_accu:.4f}, Precision: {best_precision:.4f}, F1: {best_f1:.4f}, Recall: {best_recall:.4f}, Loss: {best_loss:.4f}')
tokenizer.save_pretrained(save_dir)
