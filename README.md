# OOD_NC_project:
Out-of-Distribution Detection and Neural Collapse academic project

# Project Description:
Modern deep neural networks achieve remarkable performance on tasks where the test distribution closely matches the training distribution. However, when deployed in the real world, models routinely encounter inputs that differ significantly from what they were trained on—a situation referred to as out-of-distribution (OOD) input. A reliable classifier should not only
make correct predictions on in-distribution (ID) data, but also detect and flag OOD inputs rather than confidently misclassifying them.

In this project, we train a ResNet-18 model on CIFAR-100 as our in-distribution dataset
and use CIFAR-10 as the OOD benchmark (a near-OOD setting, since both datasets share a
similar visual domain but different label spaces). We implement and compare five OOD scoring
methods:
  • Max Softmax Probability (MSP): the classic baseline using the maximum predicted class probability.
  • Maximum Logit Score (MLS): a variant that bypasses softmax normalization.
  • Energy Score: an energy-based score derived from the log-sum-exp of logits.
  • Mahalanobis Distance: a feature-space method using per-class Gaussian models.
  • ViM (Virtual-logit Matching): a method that combines the energy score with residual feature information.

We additionally analyze the Neural Collapse (NC) phenomenon, which describes a set of
geometric regularities that emerge in the penultimate feature layer during the terminal phase
of training. Understanding NC sheds light on why certain OOD methods succeed and provides
a theoretical framework for designing better detectors.

# Setup:
```bash
python -m venv .venv # create virtual environment
.venv\Scripts\activate # or source -m .venv/bin/activate on Linux to activate the virtual environment
pip install -r requirements.txt
```
