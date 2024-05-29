import torch

def evaluate(model, dataloader, device):
    model.eval()

    # 计算 MSE 和 MAE
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

    # 计算 NDCG
    ndcg = calculate_ndcg(model, dataloader, device)

    return mse, mae, ndcg


def evaluate_every_n_epochs(model, dataloader, device, n, epoch):
    if epoch % n == 0:
        avg_loss, accuracy, mse, mae, ndcg = evaluate(model, dataloader, device)
        print(f"Epoch {epoch}: Avg Loss: {avg_loss}, Accuracy: {accuracy}, MSE: {mse}, MAE: {mae}, NDCG: {ndcg}")
        save_path = f"trained_models/model_epoch_{epoch}.pt"
        torch.save(model.state_dict(), save_path)