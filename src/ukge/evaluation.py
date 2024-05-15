import torch

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total_loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy
    # Calculate MSE and MAE
    mse = 0.0
    mae = 0.0
    num_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        mse += torch.nn.functional.mse_loss(outputs, labels, reduction='sum').item()
        mae += torch.nn.functional.l1_loss(outputs, labels, reduction='sum').item()
        num_samples += labels.size(0)

    mse /= num_samples
    mae /= num_samples

    # Calculate NDCG
    ndcg = calculate_ndcg(model, dataloader, device)

    return avg_loss, accuracy, mse, mae, ndcg

def evaluate_every_n_epochs(model, dataloader, device, n, epoch):
    if epoch % n == 0:
        avg_loss, accuracy, mse, mae, ndcg = evaluate(model, dataloader, device)
        print(f"Epoch {epoch}: Avg Loss: {avg_loss}, Accuracy: {accuracy}, MSE: {mse}, MAE: {mae}, NDCG: {ndcg}")
        save_path = f"trained_models/model_epoch_{epoch}.pt"
        torch.save(model.state_dict(), save_path)