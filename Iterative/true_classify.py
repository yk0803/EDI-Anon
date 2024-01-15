import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, f1_score, auc, precision_score, recall_score
import pandas as pd
import os
import xlrd
import openpyxl
import matplotlib.pyplot as plt


def test_images_true_classified(original_model, device, local_test_loader):
    original_model.eval()
    # Accuracy counter
    correct = 0
    correct_examples = []
    logits = []
    labels = []
    counter = 0

    # Loop over all examples in test set
    for data, target in local_test_loader:
        counter += 1

        # Send the data and label to the device
        data = data.to(device)
        target = target.to(device)
        # Forward pass the data through the model

        with torch.no_grad():
            output = original_model(data)

        final_pred = torch.argmax(output, dim=1)
        if final_pred.item() == target.item():
        	correct_examples.append(data)
        	labels.append(target)
        	logits.append(output)
        	correct += 1
    final_acc = correct/float(len(local_test_loader))

    # Return the accuracy and an adversarial example
    return final_acc, correct_examples, labels, logits

def test_images_classification(original_model, device, local_test_loader, excel_file_path, save_roc_dir):
    original_model.eval()
    # Accuracy counter
    correct = 0
    correct_examples = []
    logits = []
    labels = []
    counter = 0
    
    excel_dir = os.path.dirname(excel_file_path)

    # Extract the dataset name from the complete path of local_test_loader
    dataset_name = os.path.basename(local_test_loader.dataset.root)

    # Extract the complete path of local_test_loader
    complete_path = local_test_loader.dataset.root

    # Loop over all examples in the test set
    for data, target in tqdm(local_test_loader):
        counter += 1

        # Send the data and label to the device
        data = data.to(device)
        target = target.to(device)

        # Forward pass the data through the model
        with torch.no_grad():
            output = original_model(data)

        final_pred = torch.argmax(output, dim=1)
        if final_pred.item() == target.item():
            correct += 1
        correct_examples.append(data)
        labels.append(target)
        logits.append(output)
    final_acc = correct / float(len(local_test_loader))

    # Calculate the F1 score, precision, and recall
    all_logits = torch.cat(logits, dim=0)
    all_labels = torch.cat(labels, dim=0)
    predicted_labels = torch.argmax(all_logits, dim=1)
    f1 = f1_score(all_labels.cpu(), predicted_labels.cpu(), average='macro')
    precision = precision_score(all_labels.cpu(), predicted_labels.cpu(), average='macro')
    recall = recall_score(all_labels.cpu(), predicted_labels.cpu(), average='macro')

    # Calculate and save the ROC curve data
    all_probs = torch.nn.functional.softmax(all_logits, dim=1)
    num_classes = all_probs.shape[1]
    one_hot_labels = torch.zeros_like(all_probs).scatter_(1, all_labels.view(-1, 1), 1)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(one_hot_labels[:, i].cpu(), all_probs[:, i].cpu())
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(one_hot_labels.cpu().ravel(), all_probs.cpu().ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Save summary to Excel
    if os.path.exists(excel_file_path):
        df_summary = pd.read_excel(excel_file_path, sheet_name="Summary", engine="openpyxl")
        new_row = {"Complete Path": complete_path, "Accuracy": final_acc, "F1 Score": f1, "Precision": precision, "Recall": recall, "AUC": roc_auc["micro"]}
        df_summary = df_summary.append(new_row, ignore_index=True)
        df_summary.to_excel(excel_file_path, sheet_name="Summary", index=False)
    else:
        summary_data = {"Complete Path": [complete_path], "Accuracy": [final_acc], "F1 Score": [f1], "Precision": [precision], "Recall": [recall], "AUC": [roc_auc["micro"]]}
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(excel_file_path, sheet_name="Summary", index=False)
    
    # Plot and save the ROC curve
    plot_roc_curve(fpr, tpr, roc_auc, dataset_name, save_roc_dir)

    # Return the accuracy, F1 score, and an adversarial example
    return final_acc, correct_examples, labels, logits

def plot_roc_curve(fpr, tpr, roc_auc, dataset_name, save_dir):
    plt.figure()
    lw = 2

    # Plot macro-average ROC curve
    plt.plot(fpr['micro'], tpr['micro'], color='deeppink', linestyle=':', lw=lw,
             label='Macro-average ROC curve (area = {0:0.2f})'.format(roc_auc['micro']))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for ' + dataset_name)
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(save_dir, f'ROC_{dataset_name}.png'))
    plt.close()