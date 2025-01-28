# SimpleCLIP: 图像-文本匹配模型

`SimpleCLIP` 是一个简单的图像-文本匹配模型，结合了预训练的 ResNet50 图像编码器和 GPT-2 文本编码器，并通过投影机制将这两种模态映射到同一个嵌入空间。该模型用于计算图像和文本的相似度。

### 安装依赖

```bash
git clone https://github.com/your-username/simple-clip.git
cd simple-clip
pip install -r requirements.txt
```

确保已安装 PyTorch、Transformers、torchvision 和 Pillow 库。

## 使用方法

### 加载模型

可以实例化 `SimpleCLIP` 模型，并将其移动到 GPU 或 CPU。

```python
import torch
from simple_clip import SimpleCLIP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCLIP().to(device)
```

### 数据集

该模型使用 Flickr30K 数据集进行训练和评估。你需要提供图像和对应的字幕路径。

```python
from simple_clip import Flickr30KDataset

img_dir = '/path/to/flickr30k/images'
captions_file = '/path/to/flickr30k/captions.txt'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = Flickr30KDataset(img_dir, captions_file, transform=transform)
```

### 数据集分割

使用 `split_dataset` 函数将数据集分割为训练集和测试集：

```python
train_dataset, test_dataset = split_dataset(dataset, test_ratio=0.2)
train_dataloader = DataLoader(train_dataset, batch_size=32， shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32， shuffle=False)
```

## 训练

使用以下函数在单个 GPU 上进行训练：

```python
from simple_clip import train_with_single_gpu

train_with_single_gpu(model, train_dataloader, device)
```

训练将运行 10 个 epoch，每个 epoch 结束时会输出损失值。

## 评估

训练完成后，使用以下函数进行评估：

```python
from simple_clip import eval_with_single_gpu

eval_with_single_gpu(model, test_dataloader, device)
```

该函数将计算模型在测试集上的准确率。

## 数据集

该实现使用了 Flickr30K 数据集，包含 30,000 张图片和对应的文本描述。你可以从 [Flickr30K 数据集官网](http://shannon.cs.illinois.edu/DenotationGraph/) 下载该数据集。

数据集文件需要包含图像文件名和对应的字幕，并且每行以 tab 分隔。

### 示例：

```
full_dataset = Flickr30KDataset(img_dir="/data/wangbin/week1/flickr_30k/flickr30k-images", captions_file="/data/wangbin/week1/flickr_30k/results_20130124.token"， transform=transform)
train_dataset, eval_dataset = split_dataset(full_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=32， shuffle=True)

eval_dataloader = DataLoader(eval_dataset, batch_size=32， shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleCLIP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
    
train_with_single_gpu(model, train_dataloader, device)

# 输出

```

## 许可证

本项目采用 MIT 许可证，详细信息请见 [LICENSE](LICENSE) 文件。

---
