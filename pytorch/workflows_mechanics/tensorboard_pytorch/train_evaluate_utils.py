import torch
from tqdm import tqdm

def train_one_epoch(model, loss_fn, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    running_steps = len(data_loader)
    
    data_loader = tqdm(data_loader)
    for step, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # forward
        logits = model(images)
        
        # backward
        optimizer.zero_grad()
        loss = loss_fn(logits, labels)
        loss.backward()
        
        running_loss += loss.item()
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(loss.item(), 3))
        
    return running_loss / running_steps

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    
    correct = 0
    num_samples = len(data_loader.dataset)
    
    with torch.no_grad():
        data_loader = tqdm(data_loader)
        for step, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forward
            logits = model(images)
            _, predict_labels = torch.max(logits, dim=1)

            correct += torch.eq(predict_labels, labels).sum()
    
        acc = correct.item() / num_samples
    
    return acc

def evaluate_one(image, class_indices, model, loss_fn, device):
    '''
    params:
        image: one image, [1, channle, height, width]
    '''
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        # logits shape: (1, num_classes)
        logits = model(image)
        # 如果是一张图，就squeeze，如果是batch，就不需要
        logits = torch.squeeze(logits).cpu()
        predict = torch.softmax(logits, dim=1)
        # if want to use numpy(), the tensor must be on cpu
        predict_class = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indices[str(predict_class)],
                                                 predict[predict_class].numpy())