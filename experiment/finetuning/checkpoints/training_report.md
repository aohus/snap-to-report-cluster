# Training Report

- **Model**: MultiModalNet (EfficientNet-B3 + MetaData)
- **Embedding Dim**: 128
- **Epochs**: 5
- **Batch Size**: 16
- **Learning Rate**: 0.0001

## Loss History

- **Epoch 1**: 0.9048
- **Epoch 2**: 0.7066
- **Epoch 3**: 0.6178
- **Epoch 4**: 0.5865
- **Epoch 5**: 0.5210

## Evaluation Results (from experiment.py)

| Job ID         | NMI Score | ARI Score |
| :------------- | :-------- | :-------- |
| job_h2PJBZantA | 0.7388    | 0.3819    |
| job_Mw3mB25vjA | 0.8895    | 0.5481    |
| job_sXwMciZ7AG | 0.8968    | 0.5338    |
| job_ZI1tbhOeLs | 1.0000    | 1.0000    |

**Average NMI**: 0.8813
**Average ARI**: 0.6160
