import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import get_peft_model, PromptEncoderConfig, TaskType
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    文本数据集类，用于处理文本微调数据
    """
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 支持多种文件格式
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        elif data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))
        elif data_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(data_path)
            self.data = df.to_dict('records')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 处理不同的数据格式
        if 'input' in item and 'output' in item:
            input_text = item['input']
            output_text = item['output']
        elif 'prompt' in item and 'response' in item:
            input_text = item['prompt']
            output_text = item['response']
        elif 'question' in item and 'answer' in item:
            input_text = item['question']
            output_text = item['answer']
        else:
            # 假设第一列是输入，第二列是输出
            keys = list(item.keys())
            if len(keys) >= 2:
                input_text = str(item[keys[0]])
                output_text = str(item[keys[1]])
            else:
                raise ValueError(f"无法解析数据项: {item}")
        
        # 构造模型输入
        full_text = f"{input_text}\n{output_text}"
        
        # 编码
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # 创建labels（只对输出部分计算损失）
        input_tokens = self.tokenizer(input_text, return_tensors='pt')['input_ids'].squeeze()
        labels = input_ids.clone()
        labels[:len(input_tokens)] = -100  # 忽略输入部分的损失
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class PTuningFinetuning:
    """
    P-Tuning v2 微调实现类
    """
    
    def __init__(self, model_name: str, pre_seq_len: int = 16, prefix_projection: bool = True, encoder_hidden_size: int = 128):
        """
        初始化P-Tuning微调器
        
        Args:
            model_name: 预训练模型名称
            pre_seq_len: 前缀序列长度
            prefix_projection: 是否使用前缀投影
            encoder_hidden_size: 编码器隐藏层大小
        """
        self.model_name = model_name
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.encoder_hidden_size = encoder_hidden_size
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
        # 初始化模型和分词器
        self._init_model()
    
    def _init_model(self):
        """
        初始化模型和分词器
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            # 处理不同模型的pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # 配置P-Tuning v2参数
            peft_config = PromptEncoderConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.pre_seq_len,
                encoder_hidden_size=self.encoder_hidden_size,
                prefix_projection=self.prefix_projection
            )
            
            # 获取PEFT模型
            self.peft_model = get_peft_model(self.model, peft_config)
            logger.info("Model and PEFT configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise e
    
    def prepare_dataset(self, data_path: str, batch_size: int = 4) -> DataLoader:
        """
        准备数据集
        
        Args:
            data_path: 数据路径
            batch_size: 批次大小
            
        Returns:
            DataLoader: 数据加载器
        """
        dataset = TextDataset(data_path, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader
    
    def train(
        self,
        train_dataloader: DataLoader,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        output_dir: str = "./finetuned_model",
        log_callback=None
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_dataloader: 训练数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
            warmup_ratio: warmup比例
            output_dir: 模型保存路径
            log_callback: 日志回调函数
            
        Returns:
            Dict: 训练结果
        """
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.peft_model.to(device)
        
        # 设置优化器
        optimizer = optim.AdamW(self.peft_model.parameters(), lr=learning_rate)
        
        # 计算总步数和warmup步数
        total_steps = len(train_dataloader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        # 设置学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练过程
        self.peft_model.train()
        training_stats = []
        
        if log_callback:
            log_callback(f"开始训练: {epochs} 轮, 总步数: {total_steps}, Warmup步数: {warmup_steps}")
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            if log_callback:
                log_callback(f"开始第 {epoch+1} 轮训练...")
            
            for batch_idx, batch in enumerate(train_dataloader):
                # 将数据移到设备上
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # 前向传播
                outputs = self.peft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # 打印训练进度
                if batch_idx % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    log_msg = f"轮次: {epoch+1}/{epochs}, 批次: {batch_idx}, 损失: {loss.item():.4f}, 学习率: {current_lr:.2e}"
                    logger.info(log_msg)
                    if log_callback:
                        log_callback(log_msg)
            
            # 计算平均损失
            avg_loss = total_loss / num_batches
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"轮次 {epoch+1} 完成. 平均损失: {avg_loss:.4f}, 最终学习率: {current_lr:.2e}")
            
            if log_callback:
                log_callback(f"轮次 {epoch+1} 完成. 平均损失: {avg_loss:.4f}")
            
            # 保存训练状态
            training_stats.append({
                "epoch": epoch + 1,
                "average_loss": avg_loss
            })
        
        # 保存模型
        self.save_model(output_dir)
        
        if log_callback:
            log_callback("✅ 模型训练完成，已保存到: " + output_dir)
        
        return {
            "status": "success",
            "training_stats": training_stats,
            "output_dir": output_dir
        }
    
    def save_model(self, output_dir: str):
        """
        保存模型
        
        Args:
            output_dir: 模型保存路径
        """
        os.makedirs(output_dir, exist_ok=True)
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")


def run_ptuning_finetuning(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    pre_seq_len: int = 16,
    prefix_projection: bool = True,
    encoder_hidden_size: int = 128,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    log_callback=None
) -> Dict[str, Any]:
    """
    运行P-Tuning v2微调
    
    Args:
        model_name: 预训练模型名称
        dataset_path: 数据集路径
        output_dir: 模型保存路径
        pre_seq_len: 前缀序列长度
        prefix_projection: 是否使用前缀投影
        encoder_hidden_size: 编码器隐藏层大小
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        warmup_ratio: warmup比例
        log_callback: 日志回调函数
        
    Returns:
        Dict: 训练结果
    """
    try:
        if log_callback:
            log_callback(f"开始初始化 P-Tuning v2 微调...")
            log_callback(f"模型: {model_name}")
            log_callback(f"数据集: {dataset_path}")
            log_callback(f"前缀长度: {pre_seq_len}")
            log_callback(f"启用前缀投影: {prefix_projection}")
            log_callback(f"编码器隐藏层大小: {encoder_hidden_size}")
        
        # 初始化微调器
        finetuner = PTuningFinetuning(
            model_name=model_name,
            pre_seq_len=pre_seq_len,
            prefix_projection=prefix_projection,
            encoder_hidden_size=encoder_hidden_size
        )
        
        if log_callback:
            log_callback("模型加载完成，准备数据集...")
        
        # 准备数据集
        train_dataloader = finetuner.prepare_dataset(dataset_path, batch_size)
        
        if log_callback:
            log_callback(f"数据集准备完成，共 {len(train_dataloader.dataset)} 个样本")
            log_callback("开始训练...")
        
        # 开始训练
        result = finetuner.train(
            train_dataloader=train_dataloader,
            epochs=epochs,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            output_dir=output_dir,
            log_callback=log_callback
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        if log_callback:
            log_callback(f"❌ 微调过程中出错: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }