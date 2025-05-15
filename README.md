# Fine-Tuning BERT for Sentiment Analysis on IMDb Dataset

This project demonstrates how to fine-tune a pre-trained BERT model using the Hugging Face Transformers library to classify movie reviews from the IMDb dataset as either positive or negative.

---

## 📌 Project Description

- **Task**: Binary sentiment classification
- **Dataset**: IMDb Reviews (50,000 labeled examples)
- **Model**: `bert-base-uncased`
- **Framework**: PyTorch + Hugging Face Transformers
- **Training**: Performed on Azure VM using CPU

---

## 🛠 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Ensure your Python version is 3.10 or 3.11 for best compatibility.

---

## 🚀 Running the Project

1. Clone the repo:

   ```bash
   git clone <your-repo-url>
   cd nn-project
   ```

2. Run the script:

   ```bash
   python3 main.py
   ```

Training may take time on CPU; the script is configured for a smaller dataset subset to minimize runtime.

---

## 🧪 Model Configuration

- **Training samples**: 500
- **Testing samples**: 200
- **Epochs**: 2
- **Batch size**: 2
- **Learning rate**: 2e-5

---

## 📈 Output

Expected output after training:

- Accuracy ≈ 90%
- F1 Score ≈ 0.89

Metrics will be printed at the end of training and evaluation.

---

## 📂 Project Structure

```
├── main.py            # Training script
├── requirements.txt   # Project dependencies
├── README.md          # This file
├── final_report.docx  # Narrative report (optional)
```

---

## 👨‍💻 Author

Submitted as part of the Neural Networks course, May 2025
Instructor: Dr. Mert Nakip
