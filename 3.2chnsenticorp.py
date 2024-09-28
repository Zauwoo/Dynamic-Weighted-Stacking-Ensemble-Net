import os
import time
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
from paddlenlp.transformers import NeZhaForSequenceClassification, NeZhaTokenizer
from paddlenlp.transformers import XLNetForSequenceClassification, XLNetTokenizer
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.transformers import LinearDecayWithWarmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import LabelBinarizer

# 定义要进行分类的类别
label_list = ['0', '1']  # 0表示负面情绪，1表示正面情绪
label_map = {label: idx for idx, label in enumerate(label_list)}
print(label_map)

# 定义数据集类
class ChnSentiCorpData(Dataset):
    def __init__(self, path):
        super(ChnSentiCorpData, self).__init__()
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
    text = example[0]
    encoded_inputs = tokenizer(text=text, max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    if not is_test:
        label = np.array([example[1]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids

# 数据加载函数dataloader
def create_dataloader(dataset, mode='train', batch_size=128, tokenizer=None):
    trans_func = lambda example: convert_example(example, tokenizer=tokenizer, max_seq_length=128)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        Stack()  # labels
    ): [data for data in fn(samples)]
    
    dataset = [trans_func(example) for example in dataset]
    shuffle = True if mode == 'train' else False
    batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    return paddle.io.DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)

# 定义超参
batch_size = 128
max_seq_length = 128
learning_rate = 2e-5
epochs = 5
warmup_proportion = 0.1
weight_decay = 0.01

# 定义混合损失函数
class MixedLoss(nn.Layer):
    def __init__(self, alpha=0.5):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits, labels, probs):
        ce_loss = self.cross_entropy(logits, labels)
        kl_loss = self.kl_div(F.log_softmax(logits, axis=-1), probs)
        return self.alpha * ce_loss + (1 - self.alpha) * kl_loss

# 定义多类分类 Metric
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

@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    all_probs = []
    all_labels = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        loss = criterion(logits, labels, probs)
        losses.append(loss.numpy())
        metric.update(probs, labels)
        all_probs.extend(probs.numpy())
        all_labels.extend(labels.numpy())
    accu, precision, recall, f1 = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f, precision: %.5f, recall: %.5f, f1: %.5f" % (np.mean(losses), accu, precision, recall, f1))
    model.train()
    return accu, precision, recall, f1, np.mean(losses), np.array(all_probs), np.array(all_labels)

def train_and_evaluate(model, tokenizer, train_data_loader, dev_data_loader, epochs, learning_rate, warmup_proportion, weight_decay, model_name, fold_idx):
    num_training_steps = len(train_data_loader) * epochs
    lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])

    criterion = MixedLoss(alpha=0.5)

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
    total_train_time = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, labels = batch
            logits = model(input_ids, token_type_ids)
            probs = F.softmax(logits, axis=1)
            loss = criterion(logits, labels, probs)
            metric.update(probs, labels)
            acc, precision, recall, f1 = metric.accumulate()

            global_step += 1
            if global_step % 10 == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, f1: %.5f" % (global_step, epoch, step, loss, acc, f1))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

        epoch_train_time = time.time() - start_time
        total_train_time += epoch_train_time
        print(f"Epoch {epoch} training time: {epoch_train_time:.2f} seconds")

        accu, precision, recall, f1, loss, all_probs, all_labels = evaluate(model, criterion, metric, dev_data_loader)
        print(f'Epoch {epoch}, Accuracy: {accu:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Loss: {loss:.4f}')
        if accu > pre_accu:
            save_param_path = os.path.join(save_dir, f'{model_name}_fold{fold_idx}_epoch{epoch}_accu_{accu:.4f}_F1_{f1:.4f}_loss_{loss:.4f}_recall_{recall:.4f}.pdparams')
            paddle.save(model.state_dict(), save_param_path)
            pre_accu = accu
            best_epoch = epoch
            best_precision = precision
            best_f1 = f1
            best_recall = recall
            best_loss = loss

    print(f'Best model from Epoch {best_epoch}, Accuracy: {pre_accu:.4f}, Precision: {best_precision:.4f}, F1: {best_f1:.4f}, Recall: {best_recall:.4f}, Loss: {best_loss:.4f}')
    tokenizer.save_pretrained(save_dir)
    print(f"Total training time for fold {fold_idx}: {total_train_time:.2f} seconds")
    return pre_accu, best_precision, best_f1, best_recall, best_loss, np.array(all_probs), np.array(all_labels)

# 加载数据集
train_ds = ChnSentiCorpData('train.tsv')
test_ds = ChnSentiCorpData('test.tsv')

# 定义模型和对应的tokenizer
model_nezha = NeZhaForSequenceClassification.from_pretrained('nezha-large-wwm-chinese', num_classes=2)
tokenizer_nezha = NeZhaTokenizer.from_pretrained('nezha-large-wwm-chinese')

model_xlnet = XLNetForSequenceClassification.from_pretrained('xlnet-large-cased', num_classes=2, ignore_mismatched_sizes=True)
tokenizer_xlnet = XLNetTokenizer.from_pretrained('xlnet-large-cased')

model_ernie = ErnieForSequenceClassification.from_pretrained('ernie-1.0', num_classes=2)
tokenizer_ernie = ErnieTokenizer.from_pretrained('ernie-1.0')

# 定义6折交叉验证
kf = KFold(n_splits=6, shuffle=True, random_state=42)

# 记录最优融合模型的结果
best_meta_results = {
    'best_meta_acc': 0,
    'best_meta_f1': 0,
    'best_meta_precision': 0,
    'best_meta_recall': 0,
    'best_meta_loss': float('inf')
}

# 记录每一折的融合模型结果
fold_results = {
    'meta_acc': [],
    'meta_f1': [],
    'meta_precision': [],
    'meta_recall': [],
    'meta_loss': [],
    'train_time': []
}

# 对每个fold进行训练和验证
for fold_idx, (train_idx, dev_idx) in enumerate(kf.split(train_ds)):
    print(f"Training on fold {fold_idx + 1}...")

    # 开始计时
    fold_start_time = time.time()

    # 创建fold对应的数据集
    train_subset = paddle.io.Subset(train_ds, train_idx)
    dev_subset = paddle.io.Subset(train_ds, dev_idx)

    # 训练和评估 NeZha 模型
    train_data_loader_nezha = create_dataloader(train_subset, mode='train', batch_size=batch_size, tokenizer=tokenizer_nezha)
    dev_data_loader_nezha = create_dataloader(dev_subset, mode='dev', batch_size=batch_size, tokenizer=tokenizer_nezha)

    print('Training and evaluating NeZha...')
    acc_nezha, precision_nezha, f1_nezha, recall_nezha, loss_nezha, probs_nezha, labels_nezha = train_and_evaluate(model_nezha, tokenizer_nezha, train_data_loader_nezha, dev_data_loader_nezha, epochs, learning_rate, warmup_proportion, weight_decay, "NeZha", fold_idx)

    # 训练和评估 XLNet 模型
    train_data_loader_xlnet = create_dataloader(train_subset, mode='train', batch_size=batch_size, tokenizer=tokenizer_xlnet)
    dev_data_loader_xlnet = create_dataloader(dev_subset, mode='dev', batch_size=batch_size, tokenizer=tokenizer_xlnet)

    print('Training and evaluating XLNet...')
    acc_xlnet, precision_xlnet, f1_xlnet, recall_xlnet, loss_xlnet, probs_xlnet, _ = train_and_evaluate(model_xlnet, tokenizer_xlnet, train_data_loader_xlnet, dev_data_loader_xlnet, epochs, learning_rate, warmup_proportion, weight_decay, "XLNet", fold_idx)

    # 训练和评估 ERNIE 模型
    train_data_loader_ernie = create_dataloader(train_subset, mode='train', batch_size=batch_size, tokenizer=tokenizer_ernie)
    dev_data_loader_ernie = create_dataloader(dev_subset, mode='dev', batch_size=batch_size, tokenizer=tokenizer_ernie)

    print('Training and evaluating ERNIE...')
    acc_ernie, precision_ernie, f1_ernie, recall_ernie, loss_ernie, probs_ernie, _ = train_and_evaluate(model_ernie, tokenizer_ernie, train_data_loader_ernie, dev_data_loader_ernie, epochs, learning_rate, warmup_proportion, weight_decay, "ERNIE", fold_idx)
    
    # 使用贝叶斯优化对Stacking模型的权重进行优化
    def stacking_loss_function(weight_nezha, weight_xlnet, weight_ernie):
        final_probs = (weight_nezha * probs_nezha + weight_xlnet * probs_xlnet + weight_ernie * probs_ernie) / (weight_nezha + weight_xlnet + weight_ernie)
        final_preds = final_probs.argmax(axis=1)
        stacking_f1 = f1_score(labels_nezha, final_preds, average='weighted')
        return stacking_f1

    # 贝叶斯优化
    optimizer = BayesianOptimization(
        f=stacking_loss_function,
        pbounds={
            'weight_nezha': (0.1, 10),
            'weight_xlnet': (0.1, 10),
            'weight_ernie': (0.1, 10),
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=5, n_iter=20)

    # 获取最优权重
    best_weights = optimizer.max['params']
    best_weight_nezha = best_weights['weight_nezha']
    best_weight_xlnet = best_weights['weight_xlnet']
    best_weight_ernie = best_weights['weight_ernie']

    # 应用最优权重进行加权平均
    final_probs_weighted = (best_weight_nezha * probs_nezha + best_weight_xlnet * probs_xlnet + best_weight_ernie * probs_ernie) / (best_weight_nezha + best_weight_xlnet + best_weight_ernie)
    final_preds = final_probs_weighted.argmax(axis=1)

    # 计算Stacking模型的性能
    stacking_acc = accuracy_score(labels_nezha, final_preds)
    stacking_f1 = f1_score(labels_nezha, final_preds, average='weighted')
    stacking_precision = precision_score(labels_nezha, final_preds, average='weighted')
    stacking_recall = recall_score(labels_nezha, final_preds, average='weighted')

    # 计算Stacking模型的损失值
    lb = LabelBinarizer()
    labels_binarized = lb.fit_transform(labels_nezha)
    stacking_loss = log_loss(labels_binarized, final_probs_weighted)

    # 记录当前fold的融合模型结果
    fold_results['meta_acc'].append(stacking_acc)
    fold_results['meta_f1'].append(stacking_f1)
    fold_results['meta_precision'].append(stacking_precision)
    fold_results['meta_recall'].append(stacking_recall)
    fold_results['meta_loss'].append(stacking_loss)

    # 记录当前fold的训练时间
    fold_train_time = time.time() - fold_start_time
    fold_results['train_time'].append(fold_train_time)
    
    # 更新最优结果
    if stacking_f1 > best_meta_results['best_meta_f1']:
        best_meta_results['best_meta_acc'] = stacking_acc
        best_meta_results['best_meta_f1'] = stacking_f1
        best_meta_results['best_meta_precision'] = stacking_precision
        best_meta_results['best_meta_recall'] = stacking_recall
        best_meta_results['best_meta_loss'] = stacking_loss

    # 在每一折训练结束后，打印当前融合模型的最佳精确度、损失值和训练时间
    print(f'Fold {fold_idx + 1} results:')
    print(f'  Current ensemble model - Accuracy: {stacking_acc:.4f}, Precision: {stacking_precision:.4f}, Recall: {stacking_recall:.4f}, F1: {stacking_f1:.4f}, Loss: {stacking_loss:.4f}')
    print(f'  Best ensemble model so far - Accuracy: {best_meta_results["best_meta_acc"]:.4f}, Precision: {best_meta_results["best_meta_precision"]:.4f}, Recall: {best_meta_results["best_meta_recall"]:.4f}, F1: {best_meta_results["best_meta_f1"]:.4f}, Loss: {best_meta_results["best_meta_loss"]:.4f}')
    print(f'  Training time for this fold: {fold_train_time:.2f} seconds')

# 对所有fold的结果进行平均
final_results = {k: np.mean(v) for k, v in fold_results.items()}

print(f'Final 6-fold cross-validation results for ensemble model: {final_results}')
print(f'Best ensemble model - Accuracy: {best_meta_results["best_meta_acc"]:.4f}, Precision: {best_meta_results["best_meta_precision"]:.4f}, Recall: {best_meta_results["best_meta_recall"]:.4f}, F1: {best_meta_results["best_meta_f1"]:.4f}, Loss: {best_meta_results["best_meta_loss"]:.4f}')
