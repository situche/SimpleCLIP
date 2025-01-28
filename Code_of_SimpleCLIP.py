import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image

class SimpleCLIP(nn.Module):
    def __init__(self, dim=512):
        super(SimpleCLIP, self)。__init__()
        self.img_encoder = resnet50(pretrained=True)
        self.text_encoder = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        self.img_projection = nn.Linear(self.img_encoder.fc.out_features, dim)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, dim)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
    
    def forward(self, img, text):
        img_features = self.img_encoder(img)
        text_outputs = self.text_encoder(text, output_hidden_states=True)
        text_features = text_outputs.hidden_states[-1][:, -1, :]
        img_embedding = F.normalize(self.img_projection(img_features), dim=1)
        text_embedding = F.normalize(self.text_projection(text_features), dim=1)
        logits = torch.matmul(img_embedding, text_embedding.T) * torch.exp(self.temperature)
        return logits

class Flickr30KDataset(Dataset):
    def __init__(self, img_dir, captions_file, transform=None, max_length=64):
        self.img_dir = img_dir
        self.captions_file = captions_file
        self.transform = transform
        self.images = []
        self.captions = []
        self.max_length = max_length

        with open(captions_file, 'r') as f:
            for line in f:
                img_name, caption = line.strip().split('\t')
                img_name = img_name.split('#')[0]
                self.images.append(img_name)
                self.captions.append(caption)
        
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
   
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        caption = self.captions[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        text_inputs = self.tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
        return image, text_inputs
    
def split_dataset(dataset, test_ratio=0.2):
    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleCLIP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def train_with_single_gpu(model, train_dataloader, device):
    for epoch in range(10):
        model.train()
        running_loss = 0.0

        for images, texts in train_dataloader:
            images = images.to(device)
            input_ids = texts['input_ids'].squeeze(1).to(device)
            
            optimizer.zero_grad()

            logits = model(images, input_ids)
            labels = torch.arange(logits.shape[0], device=device)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    print(f'Training complete. Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_dataloader)}')

def eval_with_single_gpu(model, eval_dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, texts in eval_dataloader:
            images = images.to(device)
            input_ids = texts['input_ids'].squeeze(1).to(device)

            logits = model(images, input_ids)
            labels = torch.arange(logits.shape[0], device=device)

            predections = logits.argmax(dim=1)
            total_correct += (predections == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    print(f'Evaluation complete. Epoch [{epoch + 1}/10], eval accuracy: {accuracy * 100:.2f}%')
