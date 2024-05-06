import torch
from tqdm import tqdm


def validate(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in tqdm(dataloader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            total_correct += (predicted == target).sum().item()

    accuracy = 100.0 * total_correct / total_samples
    return accuracy


def evaluate(model, criterion, iterator):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in tqdm(iterator):
            input_data, target_data = batch
            if torch.cuda.is_available():
                input_data, target_data = input_data.cuda(), target_data.cuda()

            output = model(input_data)
            loss = criterion(output, target_data)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
