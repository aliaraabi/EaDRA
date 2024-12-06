
# **EaDRA: Entropy- and Distance-Regularized Attention**

This project implements the EaDRA loss function, which modifies the standard Label Smoothed Cross-Entropy Loss in Fairseq to include entropy and distance-based 
regularizers. It enhances generalization and robustness in low-resource Neural Machine Translation (NMT) tasks.

## **Installation**

1. Clone this repository:

```bash
git clone https://github.com/aliaraabi/EaDRA.git
cd EaDRA
   ```
2.Install Fairseq following the official instructions.

3.Copy the label_smoothed_cross_entropy_eadra.py file to your Fairseq installation:
```bash
cp label_smoothed_cross_entropy_eadra.py /path/to/fairseq/fairseq/criterions/
```

## **Usage**
To use this criterion in your Fairseq project:

1.Modify the Fairseq training script to specify --criterion label_smoothed_cross_entropy_eadra as the loss function.

2.Adjust the regularization coefficients in your configuration file or via command-line arguments:
```bash
--entropy-coef <value> \
--ave-entropy-coef <value> \
--sinkhorn-coef <value> \
--dec-entropy-coef <value> \
--dec-ave-entropy-coef <value> \
--dec-sinkhorn-coef <value> \
--decx-entropy-coef <value> \
--decx-ave-entropy-coef <value> \
--decx-sinkhorn-coef <value>

```

3.Train your model as usual.


## **Key Features**
- Entropy Regularization: Encourages peaked attention distributions, improving the model's focus.
- Distance Regularization: Induces preference for attending to adjacent tokens, enhancing robustness.
- Cross-Attention Regularization: Extends the entropy and distance penalties to the encoder-decoder attention mechanism.

## **Acknowledgments**
This implementation builds upon the Fairseq framework by Facebook AI Research.

## **Citation**

If you use this code in your work, please cite our paper:

Title: Entropy– and Distance-Regularized Attention Improves Low-Resource Neural Machine Translation

BibTeX:
```bash
@inproceedings{araabi-etal-2024-entropy,
    title = "Entropy{--} and Distance-Regularized Attention Improves Low-Resource Neural Machine Translation",
    author = "Araabi, Ali  and
      Niculae, Vlad  and
      Monz, Christof",
    editor = "Knowles, Rebecca  and
      Eriguchi, Akiko  and
      Goel, Shivali",
    booktitle = "Proceedings of the 16th Conference of the Association for Machine Translation in the Americas (Volume 1: Research Track)",
    month = sep,
    year = "2024",
    address = "Chicago, USA",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://aclanthology.org/2024.amta-research.13",
    pages = "140--153",
}
```


