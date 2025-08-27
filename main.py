# MyGO: Memory Yielding Generative Offline-consolidation for Lifelong Learning Systems
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset as TorchSubset
import numpy as np
import copy
from tqdm import tqdm
import os
import requests
import csv
from sklearn.feature_extraction.text import CountVectorizer

# --- CV 库 ---
import torchvision
import torchvision.transforms as transforms

# --- 全局设置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================================================================
# I. 计算机视觉 (CV) 部分
# ==========================================================================================

# --- CV 超参数 ---
CV_LR_FAST = 0.001
CV_LR_SLOW = 0.0001
CV_BATCH_SIZE = 128
CV_WAKE_EPOCHS = 5
CV_SLEEP_STEPS = 1000
CV_Z_DIM = 64
CV_NUM_TASKS = 5
CV_CLASSES_PER_TASK = 2

# --- CV 数据加载 ---
def get_split_mnist_tasks(num_tasks, classes_per_task):
    print("Preparing Split-MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    full_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    tasks = []
    for i in range(num_tasks):
        task_labels = list(range(i * classes_per_task, (i + 1) * classes_per_task))
        indices = [idx for idx, target in enumerate(full_dataset.targets) if target in task_labels]
        task_dataset = TorchSubset(full_dataset, indices)
        tasks.append(task_dataset)
        print(f"Task {i+1} has {len(task_dataset)} samples for labels {task_labels}.")
    return tasks

# --- CV 模型架构 ---
class Neocortex_Net_CV(nn.Module):
    def __init__(self):
        super(Neocortex_Net_CV, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 256)
        )
        self.classifiers = nn.ModuleDict()
    def add_task_head(self, task_id):
        self.classifiers[f'task_{task_id}'] = nn.Linear(256, CV_CLASSES_PER_TASK)
    def forward(self, x, task_id):
        features = self.features(x)
        return self.classifiers[f'task_{task_id}'](features)

class Generator_CV(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(Generator_CV, self).__init__()
        self.embed = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(z_dim + num_classes, 128), nn.ReLU(),
            nn.Linear(128, 784), nn.Tanh()
        )
    def forward(self, z, c):
        c_emb = self.embed(c)
        x = torch.cat([z, c_emb], 1)
        return self.net(x).view(-1, 1, 28, 28)

class Discriminator_CV(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator_CV, self).__init__()
        self.embed = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(784 + num_classes, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x, c):
        c_emb = self.embed(c)
        x = x.view(-1, 784)
        x = torch.cat([x, c_emb], 1)
        return self.net(x)

# ==========================================================================================
# II. 自然语言处理 (NLP) 部分
# ==========================================================================================

# --- NLP 超参数 ---
NLP_LR_FAST = 0.001
NLP_LR_SLOW = 0.0001
NLP_BATCH_SIZE = 128
NLP_WAKE_EPOCHS = 5
NLP_SLEEP_STEPS = 1000
NLP_Z_DIM = 64
NLP_NUM_TASKS = 2
NLP_CLASSES_PER_TASK = 2
NLP_EMBED_DIM = 128
NLP_HIDDEN_DIM = 256
NLP_VOCAB_MAX_FEATURES = 10000

# --- NLP 数据加载 ---
def download_agnews_data():
    data_dir = "./data/ag_news_csv"
    if os.path.exists(data_dir): return data_dir
    print("Downloading AG News data..."); os.makedirs(data_dir, exist_ok=True)
    url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
    r = requests.get(url, allow_redirects=True)
    open(os.path.join(data_dir, 'train.csv'), 'wb').write(r.content)
    print("Download complete."); return data_dir

def get_split_agnews_tasks_manual(num_tasks, classes_per_task):
    data_dir = download_agnews_data()
    texts, labels = [], []
    with open(os.path.join(data_dir, 'train.csv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(int(row[0]) - 1)
            texts.append(row[1] + " " + row[2])
    print("Building vocabulary with scikit-learn...")
    vectorizer = CountVectorizer(max_features=NLP_VOCAB_MAX_FEATURES, stop_words='english')
    vectorizer.fit(texts); vocab = vectorizer.vocabulary_
    vocab['<unk>'] = len(vocab); unk_token_id = vocab['<unk>']
    vocab_size = len(vocab); print(f"Vocabulary size: {vocab_size}")
    def text_pipeline(text):
        tokens = vectorizer.build_analyzer()(text.lower())
        return [vocab.get(token, unk_token_id) for token in tokens]
    full_data = [{'text': text, 'label': label} for text, label in zip(texts, labels)]
    tasks = []
    for i in range(num_tasks):
        task_labels = list(range(i * classes_per_task, (i + 1) * classes_per_task))
        task_data = [d for d in full_data if d['label'] in task_labels]
        tasks.append(task_data)
        print(f"Task {i+1} has {len(task_data)} samples for labels {task_labels}.")
    return tasks, text_pipeline, vocab_size

def collate_batch_nlp(batch, text_pipeline):
    label_list, text_list, offsets = [], [], [0]
    for item in batch:
        _label, _text = item['label'], item['text']
        label_list.append(_label)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text); offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return text_list.to(DEVICE), offsets.to(DEVICE), label_list.to(DEVICE)

# --- NLP 模型架构 ---
class Neocortex_Net_NLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Neocortex_Net_NLP, self).__init__()
        self.features = nn.Sequential(
            nn.EmbeddingBag(vocab_size, embed_dim, sparse=False),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifiers = nn.ModuleDict()
    def add_task_head(self, task_id):
        self.classifiers[f'task_{task_id}'] = nn.Linear(NLP_HIDDEN_DIM, NLP_CLASSES_PER_TASK)
    def forward(self, text, offsets, task_id):
        features = self.features[0](text, offsets)
        features = self.features[1:](features)
        return self.classifiers[f'task_{task_id}'](features)

class Generator_NLP(nn.Module):
    def __init__(self, z_dim, embed_dim, num_classes):
        super(Generator_NLP, self).__init__()
        self.embed = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(z_dim + num_classes, 128), nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
    def forward(self, z, c):
        c_emb = self.embed(c); x = torch.cat([z, c_emb], 1); return self.net(x)

class Discriminator_NLP(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(Discriminator_NLP, self).__init__()
        self.embed = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + num_classes, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x_feat, c):
        c_emb = self.embed(c); x = torch.cat([x_feat, c_emb], 1); return self.net(x)

# ==========================================================================================
# III. 通用 MyGO 框架
# ==========================================================================================
# (这部分代码与之前版本相同，为了简洁在此折叠。实际运行时需要完整代码。)
class MyGO_Manager:
    def __init__(self, config):
        self.config = config
        self.M_ctx = None
        self.past_G_mems = []
        self.seen_tasks_data = []
        self.text_pipeline = None
        self.vocab_size = None

    def _prepare_data(self):
        if self.config['domain'] == 'cv':
            return get_split_mnist_tasks(self.config['num_tasks'], self.config['classes_per_task'])
        elif self.config['domain'] == 'nlp':
            tasks, self.text_pipeline, self.vocab_size = get_split_agnews_tasks_manual(self.config['num_tasks'], self.config['classes_per_task'])
            return tasks
        else:
            raise ValueError("Domain must be 'cv' or 'nlp'")

    def _initialize_models(self):
        if self.config['domain'] == 'cv':
            self.M_ctx = Neocortex_Net_CV().to(DEVICE)
        elif self.config['domain'] == 'nlp':
            self.M_ctx = Neocortex_Net_NLP(self.vocab_size, NLP_EMBED_DIM, NLP_HIDDEN_DIM).to(DEVICE)

    def _wake_phase(self, task_id, task_loader, classes_in_task):
        print(f"  Wake Phase for Task {task_id+1}...");
        for param in self.M_ctx.features.parameters(): param.requires_grad = False
        current_head = self.M_ctx.classifiers[f'task_{task_id}']; current_head.train()
        if self.config['domain'] == 'cv':
            G_mem = Generator_CV(CV_Z_DIM, len(classes_in_task)).to(DEVICE)
            D_mem = Discriminator_CV(len(classes_in_task)).to(DEVICE)
        else:
            G_mem = Generator_NLP(NLP_Z_DIM, NLP_EMBED_DIM, len(classes_in_task)).to(DEVICE)
            D_mem = Discriminator_NLP(NLP_EMBED_DIM, len(classes_in_task)).to(DEVICE)
        optim_hpc = optim.Adam(current_head.parameters(), lr=self.config['lr_fast'])
        optim_G = optim.Adam(G_mem.parameters(), lr=self.config['lr_fast'])
        optim_D = optim.Adam(D_mem.parameters(), lr=self.config['lr_fast'])
        criterion_hpc = nn.CrossEntropyLoss(); criterion_gan = nn.BCELoss()
        for epoch in range(self.config['wake_epochs']):
            print(f"    Epoch {epoch+1}/{self.config['wake_epochs']}")
            for batch in tqdm(task_loader):
                if self.config['domain'] == 'cv':
                    data, labels = batch; data, labels = data.to(DEVICE), labels.to(DEVICE)
                else:
                    text, offsets, labels = batch; data = (text, offsets)
                local_labels = labels - min(classes_in_task)
                optim_hpc.zero_grad()
                if self.config['domain'] == 'cv':
                    with torch.no_grad(): features = self.M_ctx.features(data)
                    outputs = current_head(features)
                else:
                    with torch.no_grad(): features = self.M_ctx.features[0](data[0], data[1])
                    outputs = current_head(self.M_ctx.features[1:](features))
                loss_hpc = criterion_hpc(outputs, local_labels); loss_hpc.backward(); optim_hpc.step()
                optim_D.zero_grad()
                real_target = torch.ones(labels.size(0), 1).to(DEVICE)
                fake_target = torch.zeros(labels.size(0), 1).to(DEVICE)
                if self.config['domain'] == 'cv':
                    real_loss = criterion_gan(D_mem(data, local_labels), real_target)
                    z = torch.randn(labels.size(0), CV_Z_DIM).to(DEVICE)
                    fake_data = G_mem(z, local_labels)
                    fake_loss = criterion_gan(D_mem(fake_data.detach(), local_labels), fake_target)
                else:
                    real_loss = criterion_gan(D_mem(features.detach(), local_labels), real_target)
                    z = torch.randn(labels.size(0), NLP_Z_DIM).to(DEVICE)
                    fake_data = G_mem(z, local_labels)
                    fake_loss = criterion_gan(D_mem(fake_data.detach(), local_labels), fake_target)
                loss_D = (real_loss + fake_loss) / 2; loss_D.backward(); optim_D.step()
                optim_G.zero_grad()
                if self.config['domain'] == 'cv': z = torch.randn(labels.size(0), CV_Z_DIM).to(DEVICE); fake_data = G_mem(z, local_labels)
                else: z = torch.randn(labels.size(0), NLP_Z_DIM).to(DEVICE); fake_data = G_mem(z, local_labels)
                loss_G = criterion_gan(D_mem(fake_data, local_labels), real_target); loss_G.backward(); optim_G.step()
        print(f"  Wake Phase for Task {task_id+1} complete."); return current_head.state_dict(), G_mem

    def _sleep_phase(self, task_id, M_hpc_state_dict, current_G_mem):
        print("  Sleep Phase starting..."); self.M_ctx.train()
        optim_ctx = optim.Adam(self.M_ctx.parameters(), lr=self.config['lr_slow'])
        teacher_M_ctx = copy.deepcopy(self.M_ctx)
        teacher_M_ctx.classifiers[f'task_{task_id}'].load_state_dict(M_hpc_state_dict); teacher_M_ctx.eval()
        all_G_mems = self.past_G_mems + [current_G_mem]
        for step in tqdm(range(self.config['sleep_steps'])):
            optim_ctx.zero_grad()
            dream_task_id = np.random.randint(0, len(all_G_mems))
            G_mem = all_G_mems[dream_task_id]
            local_labels = torch.randint(0, self.config['classes_per_task'], (self.config['batch_size'],)).to(DEVICE)
            if self.config['domain'] == 'cv':
                z = torch.randn(self.config['batch_size'], CV_Z_DIM).to(DEVICE)
                pseudo_data = G_mem(z, local_labels)
                with torch.no_grad(): teacher_logits = teacher_M_ctx(pseudo_data, dream_task_id)
                student_logits = self.M_ctx(pseudo_data, dream_task_id)
            else:
                z = torch.randn(self.config['batch_size'], NLP_Z_DIM).to(DEVICE)
                pseudo_features = G_mem(z, local_labels)
                with torch.no_grad(): teacher_logits = teacher_M_ctx.classifiers[f'task_{dream_task_id}'](teacher_M_ctx.features[1:](pseudo_features))
                student_logits = self.M_ctx.classifiers[f'task_{dream_task_id}'](self.M_ctx.features[1:](pseudo_features))
            loss = nn.MSELoss()(student_logits, teacher_logits.detach()); loss.backward(); optim_ctx.step()
        print("  Sleep Phase complete.")

    def _evaluate(self):
        self.M_ctx.eval(); accuracies = []
        for i, task_data in enumerate(self.seen_tasks_data):
            if self.config['domain'] == 'cv':
                task_loader = DataLoader(task_data, batch_size=self.config['batch_size'], shuffle=False)
            else:
                task_loader = DataLoader(task_data, batch_size=self.config['batch_size'], shuffle=False, collate_fn=lambda b: collate_batch_nlp(b, self.text_pipeline))
            correct, total = 0, 0
            classes_in_task = list(range(i * self.config['classes_per_task'], (i + 1) * self.config['classes_per_task']))
            with torch.no_grad():
                for batch in task_loader:
                    if self.config['domain'] == 'cv':
                        data, labels = batch; data, labels = data.to(DEVICE), labels.to(DEVICE)
                        outputs = self.M_ctx(data, task_id=i)
                    else:
                        text, offsets, labels = batch
                        outputs = self.M_ctx(text, offsets, task_id=i)
                    _, predicted = torch.max(outputs.data, 1); total += labels.size(0)
                    correct += (predicted == (labels - min(classes_in_task))).sum().item()
            accuracies.append(100 * correct / total)
        return accuracies

    def run(self):
        print(f"\n{'='*20} Running MyGO Experiment for {self.config['domain'].upper()} {'='*20}")
        tasks = self._prepare_data(); self._initialize_models()
        for task_id, task_data in enumerate(tasks):
            print(f"\n--- Starting Task {task_id+1}/{self.config['num_tasks']} ---")
            self.seen_tasks_data.append(task_data)
            if self.config['domain'] == 'cv':
                task_loader = DataLoader(task_data, batch_size=self.config['batch_size'], shuffle=True)
            else:
                task_loader = DataLoader(task_data, batch_size=self.config['batch_size'], shuffle=True, collate_fn=lambda b: collate_batch_nlp(b, self.text_pipeline))
            self.M_ctx.add_task_head(task_id); self.M_ctx.to(DEVICE)
            classes_in_task = list(range(task_id * self.config['classes_per_task'], (task_id + 1) * self.config['classes_per_task']))
            M_hpc_state_dict, current_G_mem = self._wake_phase(task_id, task_loader, classes_in_task)
            self._sleep_phase(task_id, M_hpc_state_dict, current_G_mem)
            self.past_G_mems.append(current_G_mem)
            accuracies = self._evaluate()
            print(f"Accuracies after Task {task_id+1}: {[f'{acc:.2f}%' for acc in accuracies]}")
            print(f"Average Accuracy: {np.mean(accuracies):.2f}%")

# ==========================================================================================
# IV. 顺序微调 (Sequential Finetuning) 基准
# ==========================================================================================
class Finetuning_Manager:
    """管理顺序微调基准测试的类"""
    def __init__(self, config):
        self.config = config
        self.model = None
        self.seen_tasks_data = []
        self.text_pipeline = None
        self.vocab_size = None

    def _prepare_data(self):
        # 复用 MyGO 的数据准备函数
        if self.config['domain'] == 'cv':
            return get_split_mnist_tasks(self.config['num_tasks'], self.config['classes_per_task'])
        elif self.config['domain'] == 'nlp':
            tasks, self.text_pipeline, self.vocab_size = get_split_agnews_tasks_manual(self.config['num_tasks'], self.config['classes_per_task'])
            return tasks

    def _initialize_models(self):
        # 使用与 MyGO 相同的特征提取器，但只有一个分类头
        if self.config['domain'] == 'cv':
            base_model = Neocortex_Net_CV()
            self.model = nn.Sequential(base_model.features, nn.Linear(256, self.config['classes_per_task'])).to(DEVICE)
        elif self.config['domain'] == 'nlp':
            base_model = Neocortex_Net_NLP(self.vocab_size, NLP_EMBED_DIM, NLP_HIDDEN_DIM)
            self.model = nn.Sequential(base_model.features, nn.Linear(NLP_HIDDEN_DIM, self.config['classes_per_task'])).to(DEVICE)

    def _evaluate(self, current_task_id):
        self.model.eval()
        accuracies = []
        for i, task_data in enumerate(self.seen_tasks_data):
            if i > current_task_id: continue # 不评估还没见过的任务
            if self.config['domain'] == 'cv':
                task_loader = DataLoader(task_data, batch_size=self.config['batch_size'], shuffle=False)
            else:
                task_loader = DataLoader(task_data, batch_size=self.config['batch_size'], shuffle=False, collate_fn=lambda b: collate_batch_nlp(b, self.text_pipeline))
            
            correct, total = 0, 0
            classes_in_task = list(range(i * self.config['classes_per_task'], (i + 1) * self.config['classes_per_task']))
            with torch.no_grad():
                for batch in task_loader:
                    if self.config['domain'] == 'cv':
                        data, labels = batch; data, labels = data.to(DEVICE), labels.to(DEVICE)
                        outputs = self.model(data)
                    else:
                        text, offsets, labels = batch
                        # 手动通过 Sequential 模型
                        features = self.model[0][0](text, offsets)
                        features = self.model[0][1:](features)
                        outputs = self.model[1](features)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == (labels - min(classes_in_task))).sum().item()
            accuracies.append(100 * correct / total)
        return accuracies

    def run(self):
        print(f"\n{'='*20} Running Finetuning Baseline for {self.config['domain'].upper()} {'='*20}")
        tasks = self._prepare_data()
        self._initialize_models()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr_fast'])
        criterion = nn.CrossEntropyLoss()

        for task_id, task_data in enumerate(tasks):
            print(f"\n--- Training on Task {task_id+1}/{self.config['num_tasks']} ---")
            self.seen_tasks_data.append(task_data)
            
            if self.config['domain'] == 'cv':
                task_loader = DataLoader(task_data, batch_size=self.config['batch_size'], shuffle=True)
            else:
                task_loader = DataLoader(task_data, batch_size=self.config['batch_size'], shuffle=True, collate_fn=lambda b: collate_batch_nlp(b, self.text_pipeline))
            
            self.model.train()
            for epoch in range(self.config['wake_epochs']):
                print(f"    Epoch {epoch+1}/{self.config['wake_epochs']}")
                for batch in tqdm(task_loader):
                    optimizer.zero_grad()
                    if self.config['domain'] == 'cv':
                        data, labels = batch; data, labels = data.to(DEVICE), labels.to(DEVICE)
                        outputs = self.model(data)
                    else:
                        text, offsets, labels = batch
                        features = self.model[0][0](text, offsets)
                        features = self.model[0][1:](features)
                        outputs = self.model[1](features)

                    local_labels = labels - (task_id * self.config['classes_per_task'])
                    loss = criterion(outputs, local_labels)
                    loss.backward()
                    optimizer.step()

            accuracies = self._evaluate(task_id)
            print(f"Accuracies after Task {task_id+1}: {[f'{acc:.2f}%' for acc in accuracies]}")
            print(f"Average Accuracy: {np.mean(accuracies):.2f}%")

# ==========================================================================================
# V. 主执行流程
# ==========================================================================================
if __name__ == "__main__":
    cv_config = { 'domain': 'cv', 'lr_fast': CV_LR_FAST, 'lr_slow': CV_LR_SLOW, 'batch_size': CV_BATCH_SIZE, 'wake_epochs': CV_WAKE_EPOCHS, 'sleep_steps': CV_SLEEP_STEPS, 'z_dim': CV_Z_DIM, 'num_tasks': CV_NUM_TASKS, 'classes_per_task': CV_CLASSES_PER_TASK, }
    nlp_config = { 'domain': 'nlp', 'lr_fast': NLP_LR_FAST, 'lr_slow': NLP_LR_SLOW, 'batch_size': NLP_BATCH_SIZE, 'wake_epochs': NLP_WAKE_EPOCHS, 'sleep_steps': NLP_SLEEP_STEPS, 'z_dim': NLP_Z_DIM, 'num_tasks': NLP_NUM_TASKS, 'classes_per_task': NLP_CLASSES_PER_TASK, }
    
    mygo_cv_experiment = MyGO_Manager(cv_config)
    mygo_cv_experiment.run()
    
    print("\n\n" + "#"*80 + "\n\n")

    mygo_nlp_experiment = MyGO_Manager(nlp_config)
    mygo_nlp_experiment.run()
    
    print("\n\n" + "#"*80 + "\n\n")

    finetuning_cv_experiment = Finetuning_Manager(cv_config)
    finetuning_cv_experiment.run()
    
    print("\n\n" + "#"*80 + "\n\n")

    finetuning_nlp_experiment = Finetuning_Manager(nlp_config)
    finetuning_nlp_experiment.run()
