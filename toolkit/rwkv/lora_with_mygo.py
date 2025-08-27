import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
from tqdm.auto import tqdm
import os
import gc
from rwkv.model import RWKV as RwkvModel
from transformers import AutoTokenizer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' for CUDA vRAM > 6GB, '0' for < 6GB

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
print(f"Using device: {DEVICE}, dtype: {DTYPE}")

# ==========================================================================================
# I. 超参数
# ==========================================================================================
MODEL_NAME = "RWKV/rwkv-4-169m-pile"
MODEL_PATH = "rwkv-4-169m-pile.pth"

# 学习率和训练参数
LR_FAST = 1e-4       # 唤醒阶段学习率
LR_SLOW = 3e-5       # 睡眠阶段学习
BATCH_SIZE = 16      # 根据你的显存调整
WAKE_EPOCHS = 3      # 唤醒阶段对新任务的学习轮数
SLEEP_STEPS = 200    # 睡眠阶段的知识蒸馏步数
Z_DIM = 64           # 生成器噪声维度
NUM_TASKS = 2        # 任务数量
CLASSES_PER_TASK = 2 # 每个任务的类别数量
MAX_LENGTH = 128     # Tokenizer 最大长度

# LoRA 配置
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# RWKV v4 的 LoRA 目标模块
LORA_TARGET_MODULES = ["key", "value", "receptance", "output"]

# ==========================================================================================
# II. 数据加载与预处理
# ==========================================================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_split_agnews_tasks(tokenizer, num_tasks=NUM_TASKS, classes_per_task=CLASSES_PER_TASK):
    print("Preparing Split AG News dataset...")
    dataset = load_dataset("ag_news", split='train')
    
    texts = [item['text'] for item in dataset]
    labels = [item['label'] for item in dataset]
    
    tasks = []
    for i in range(num_tasks):
        task_labels_map = list(range(i * classes_per_task, (i + 1) * classes_per_task))
        
        task_indices = [idx for idx, label in enumerate(labels) if label in task_labels_map]
        
        # 为了演示，每个任务只取一小部分数据
        task_indices = task_indices[:1000]

        task_texts = [texts[j] for j in task_indices]
        task_labels = [labels[j] for j in task_indices]

        task_dataset = TextDataset(task_texts, task_labels, tokenizer, MAX_LENGTH)
        tasks.append(task_dataset)
        print(f"Task {i+1} has {len(task_dataset)} samples for labels {task_labels_map}.")
    return tasks

# ==========================================================================================
# III. 模型架构
# ==========================================================================================
class RWKV_LoRA_Classification(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # 使用官方 rwkv 包加载模型
        self.rwkv_base = RwkvModel(model=model_path, strategy=f'cuda {DTYPE}')
        self.hidden_dim = self.rwkv_base.args.n_embd

        # 冻结基础模型参数，并应用 LoRA
        self._prepare_for_lora()
        
        # 为不同任务创建分类头
        self.classifiers = nn.ModuleDict()

    def _prepare_for_lora(self):
        # 冻结所有原始参数
        for param in self.rwkv_base.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM" # 即使是分类，也作用于语言模型骨干
        )
        # 将 LoRA 适配器应用到模型上
        # 注意：这里我们只应用一次，后续将训练同一套 LoRA 权重
        self.rwkv_peft = get_peft_model(self.rwkv_base, lora_config)
        print("\nPEFT Model Structure:")
        self.rwkv_peft.print_trainable_parameters()

    def add_task_head(self, task_id, classes_per_task):
        task_key = f'task_{task_id}'
        self.classifiers[task_key] = nn.Linear(self.hidden_dim, classes_per_task)

    def forward(self, input_ids, task_id):
        args = self.rwkv_peft.args
        x = self.rwkv_peft.w.emb(input_ids)
        for block in self.rwkv_peft.blocks:
            x = block(x)
        x = self.rwkv_peft.w.ln_f(x)

        # 取序列最后一个 token 的特征作为句子表示
        features = x[:, -1, :]
        
        task_key = f'task_{task_id}'
        logits = self.classifiers[task_key](features)
        return logits, features

# 生成器和判别器保持不变
class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(num_classes, 16)
        self.net = nn.Sequential(
            nn.Linear(z_dim + 16, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, hidden_dim),
        )
    def forward(self, z, c):
        c_emb = self.embed(c); x = torch.cat([z, c_emb], 1); return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(num_classes, 16)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 16, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x_feat, c):
        c_emb = self.embed(c); x = torch.cat([x_feat, c_emb], 1); return self.net(x)

# ==========================================================================================
# IV. MyGO!!!!!
# ==========================================================================================
class MyGO_LoRA_Manager:
    def __init__(self, config):
        self.config = config
        
        if not os.path.exists(MODEL_PATH):
            from huggingface_hub import hf_hub_download
            print(f"Downloading model weights for {MODEL_NAME}...")
            hf_hub_download(repo_id=MODEL_NAME, filename=MODEL_PATH, local_dir='.', local_dir_use_symlinks=False)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left' # 对于CausalLM，padding在左边更安全

        # 核心模型，只实例化一次
        self.M_ctx = RWKV_LoRA_Classification(MODEL_PATH).to(DEVICE)
        
        self.past_G_mems = []
        self.seen_tasks_data = []

    def _wake_phase(self, task_id, task_loader, classes_in_task):
        print(f"  Wake Phase for Task {task_id+1}...");
        task_key = f'task_{task_id}'
        
        self.M_ctx.train() # 设置整个模型为训练模式（会激活 LoRA 和分类头）
        current_classifier = self.M_ctx.classifiers[task_key]

        hidden_dim = self.M_ctx.hidden_dim
        num_classes_in_task = len(classes_in_task)

        G_mem = Generator(Z_DIM, hidden_dim, num_classes_in_task).to(DEVICE)
        D_mem = Discriminator(hidden_dim, num_classes_in_task).to(DEVICE)

        # 优化器包含 LoRA 参数和当前任务的分类头
        trainable_params = list(self.M_ctx.rwkv_peft.parameters()) + list(current_classifier.parameters())
        optim_task = optim.AdamW(trainable_params, lr=LR_FAST)
        optim_G = optim.AdamW(G_mem.parameters(), lr=LR_FAST)
        optim_D = optim.AdamW(D_mem.parameters(), lr=LR_FAST)

        criterion_ce = nn.CrossEntropyLoss(); criterion_gan = nn.BCELoss()

        for epoch in range(WAKE_EPOCHS):
            print(f"    Epoch {epoch+1}/{WAKE_EPOCHS}")
            for batch in tqdm(task_loader, leave=False):
                input_ids = batch['input_ids'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                local_labels = labels - min(classes_in_task)

                # --- 1. 训练任务专家 (LoRA + Classifier) ---
                optim_task.zero_grad()
                logits, features = self.M_ctx(input_ids, task_id)
                loss_task = criterion_ce(logits, local_labels)
                loss_task.backward()
                optim_task.step()

                # --- 2. 训练 GAN ---
                real_features = features.detach()
                optim_D.zero_grad()
                real_target = torch.ones(labels.size(0), 1, device=DEVICE)
                fake_target = torch.zeros(labels.size(0), 1, device=DEVICE)
                
                real_loss = criterion_gan(D_mem(real_features, local_labels), real_target)
                z = torch.randn(labels.size(0), Z_DIM, device=DEVICE)
                fake_features = G_mem(z, local_labels)
                fake_loss = criterion_gan(D_mem(fake_features.detach(), local_labels), fake_target)
                loss_D = (real_loss + fake_loss) / 2
                loss_D.backward()
                optim_D.step()

                optim_G.zero_grad()
                z = torch.randn(labels.size(0), Z_DIM, device=DEVICE)
                fake_features = G_mem(z, local_labels)
                loss_G = criterion_gan(D_mem(fake_features, local_labels), real_target)
                loss_G.backward()
                optim_G.step()
        
        print(f"  Wake Phase for Task {task_id+1} complete.")
        return G_mem

    def _sleep_phase(self, task_id):
        if task_id == 0:
            print("  Skipping Sleep Phase for the first task.")
            return

        print("  Sleep Phase starting...");
        
        # 教师模型是唤醒阶段刚结束的模型状态的深拷贝
        teacher_M_ctx = copy.deepcopy(self.M_ctx)
        teacher_M_ctx.eval()

        # 学生模型是当前 M_ctx，我们将继续训练它的 LoRA 权重和所有分类头
        self.M_ctx.train()
        
        # 优化器现在包括 LoRA 和所有见过的分类头
        trainable_params = list(self.M_ctx.rwkv_peft.parameters()) + list(self.M_ctx.classifiers.parameters())
        optim_consolidate = optim.AdamW(trainable_params, lr=LR_SLOW)
        
        criterion_distill = nn.MSELoss()

        for step in tqdm(range(SLEEP_STEPS), leave=False):
            optim_consolidate.zero_grad()
            
            # 随机从过去的任务中做梦
            dream_task_id = np.random.randint(0, task_id + 1)
            G_mem = self.past_G_mems[dream_task_id] if dream_task_id < task_id else self.current_G_mem
            
            # 生成伪特征
            local_labels = torch.randint(0, CLASSES_PER_TASK, (BATCH_SIZE,), device=DEVICE)
            z = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)
            pseudo_features = G_mem(z, local_labels)

            # 教师产生 soft labels
            with torch.no_grad():
                teacher_logits, _ = teacher_M_ctx(pseudo_features, task_id=dream_task_id) # 这里有个小bug, forward第一个参数是input_ids
                # 修正：直接用分类头
                teacher_logits = teacher_M_ctx.classifiers[f'task_{dream_task_id}'](pseudo_features)
            
            # 学生试图匹配
            student_logits = self.M_ctx.classifiers[f'task_{dream_task_id}'](pseudo_features)
            
            loss = criterion_distill(student_logits, teacher_logits.detach())
            loss.backward()
            optim_consolidate.step()

        print("  Sleep Phase complete.")

    def _evaluate(self, current_task_id):
        self.M_ctx.eval()
        accuracies = []
        
        for i in range(current_task_id + 1):
            task_data = self.seen_tasks_data[i]
            # 降低评估时的批量大小以防显存不足
            task_loader = DataLoader(task_data, batch_size=BATCH_SIZE * 2)
            correct, total = 0, 0
            classes_in_task = list(range(i * CLASSES_PER_TASK, (i + 1) * CLASSES_PER_TASK))
            
            with torch.no_grad():
                for batch in task_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    labels = batch['labels'].to(DEVICE)
                    
                    logits, _ = self.M_ctx(input_ids, task_id=i)
                    
                    _, predicted = torch.max(logits.data, 1)
                    local_labels = labels - min(classes_in_task)
                    
                    total += labels.size(0)
                    correct += (predicted == local_labels).sum().item()
            
            acc = 100 * correct / total
            accuracies.append(acc)
        return accuracies

    def run(self):
        print(f"\n{'='*20} Running MyGO-LoRA Experiment for RWKV {'='*20}")
        tasks = get_split_agnews_tasks(self.tokenizer)
        
        for task_id, task_data in enumerate(tasks):
            print(f"\n--- Starting Task {task_id+1}/{NUM_TASKS} ---")
            self.seen_tasks_data.append(task_data)
            
            # 1. 为新任务添加分类头
            self.M_ctx.add_task_head(task_id, CLASSES_PER_TASK)
            self.M_ctx.to(DEVICE)
            
            task_loader = DataLoader(task_data, batch_size=BATCH_SIZE, shuffle=True)
            classes_in_task = list(range(task_id * CLASSES_PER_TASK, (task_id + 1) * CLASSES_PER_TASK))

            # 2. 唤醒阶段：学习新任务
            self.current_G_mem = self._wake_phase(task_id, task_loader, classes_in_task)
            
            # 3. 睡眠阶段：整合新旧知识到同一套 LoRA 权重中
            self._sleep_phase(task_id)
            
            self.past_G_mems.append(self.current_G_mem)

            # 4. 评估所有已见任务
            accuracies = self._evaluate(task_id)
            print(f"Accuracies after Task {task_id+1}: {[f'{acc:.2f}%' for acc in accuracies]}")
            print(f"Average Accuracy: {np.mean(accuracies):.2f}%")
            
            # 清理显存
            gc.collect()
            torch.cuda.empty_cache()

# ==========================================================================================
# V. 主执行流程
# ==========================================================================================
if __name__ == "__main__":
    mygo_lora_experiment = MyGO_LoRA_Manager({})
    mygo_lora_experiment.run()
