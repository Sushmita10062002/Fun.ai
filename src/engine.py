import torch
import config
from tqdm import tqdm

def train_fn(data_loader, model, optimizer, device, scheduler, epoch):
    model.train()
    batch_losses = []
    final_loss = 0
    for i, data in enumerate(tqdm(data_loader, total = len(data_loader))):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        outputs = model(**data)
        loss, logits = outputs[:2]
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (i + 1) % 500 == 0:
            print('Epoch [{}/{}], bi[{}/{}], Loss: {:.4f}'
                   .format(epoch+1, config.EPOCHS, i+1,len(data_loader), loss.item()))

            batch_losses.append(loss.item())
        final_loss += loss.item()
    return (final_loss/len(data_loader)) , batch_losses
