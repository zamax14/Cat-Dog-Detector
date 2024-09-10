import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from neural_network import CNN
from torchmetrics import Accuracy, Precision, Recall

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

val_dataset = datasets.ImageFolder(root='dataset/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


model = CNN(in_chann=3, num_class=2)
model.load_state_dict(torch.load('model.pt', weights_only=True))
model.eval()

predictions, targets = [], []
with torch.no_grad():
    for _, data in tqdm(enumerate(val_loader)):
        images, labels = data
        outputs = model(images)
        preds = torch.argmax(outputs, dim=-1)
        predictions+=preds.tolist()
        targets+=labels.tolist()

accuracy_fn = Accuracy(task="multiclass", num_classes=2)
accuracy_fn(torch.tensor(predictions), torch.tensor(targets))
accuracy = accuracy_fn.compute().item()

precision_fn = Precision(task="multiclass", num_classes=2, average=None)
precision_fn(torch.tensor(predictions), torch.tensor(targets))
precision = precision_fn.compute().tolist()

recall_fn = Recall(task="multiclass", num_classes=2, average=None)
recall_fn(torch.tensor(predictions), torch.tensor(targets))
recall = recall_fn.compute().tolist()

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')