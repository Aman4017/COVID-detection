# COVID-detection

Utilized an X-ray image dataset of COVID patients to train multiple models and analyze their accuracy. The models were evaluated for their ability to accurately classify COVID-positive cases and distinguish them from other respiratory conditions, and further investigation was conducted to identify performance-enhancing patterns.

## X-ray Images

![x-ray images](https://github.com/Aman4017/COVID-detection/assets/108785124/23f74947-cb86-476e-a600-47f8588febf5)


## Models
### SVM
```bash
  Accuracy: 0.84
  Precision: 0.70
  Recall: 0.66
  F1 Score: 0.68
  Classification Report:
                 precision    recall  f1-score   support
  
        Normal       0.88      0.90      0.89      2039
         COVID       0.70      0.66      0.68       723
  
      accuracy                           0.84      2762
     macro avg       0.79      0.78      0.79      2762
  weighted avg       0.84      0.84      0.84      2762
```

### XgBoost

```bash
  Accuracy: 0.94
  Precision: 0.93
  Recall: 0.85
  F1 Score: 0.89
  Classification Report:
                 precision    recall  f1-score   support
  
        Normal       0.95      0.98      0.96      2039
         COVID       0.93      0.85      0.89       723
  
      accuracy                           0.94      2762
     macro avg       0.94      0.91      0.93      2762
  weighted avg       0.94      0.94      0.94      2762
```


### Random Forest


```bash
  Accuracy: 0.92
  Precision: 0.92
  Recall: 0.78
  F1 Score: 0.84
  Classification Report:
                 precision    recall  f1-score   support
  
        Normal       0.93      0.97      0.95      2039
         COVID       0.92      0.78      0.84       723
  
      accuracy                           0.92      2762
     macro avg       0.92      0.88      0.90      2762
  weighted avg       0.92      0.92      0.92      2762
```

### MLP

```bash
  Accuracy: 0.91
  Precision: 0.81
  Recall: 0.84
  F1 Score: 0.83
  Classification Report:
                 precision    recall  f1-score   support
  
        Normal       0.94      0.93      0.94      2039
         COVID       0.81      0.84      0.83       723
  
      accuracy                           0.91      2762
     macro avg       0.88      0.89      0.88      2762
  weighted avg       0.91      0.91      0.91      2762
```

### Naive Bayes

```bash
  Accuracy: 0.71
  Precision: 0.46
  Recall: 0.69
  F1 Score: 0.56
  Classification Report:
                 precision    recall  f1-score   support
  
        Normal       0.87      0.72      0.79      2039
         COVID       0.46      0.69      0.56       723
  
      accuracy                           0.71      2762
     macro avg       0.67      0.70      0.67      2762
  weighted avg       0.76      0.71      0.73      2762
```
