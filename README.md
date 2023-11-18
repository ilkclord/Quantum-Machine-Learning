# Quantum-Machine-Learning
hackathon 


QCNet(
  (conv1): Conv2d(6, 9, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(9, 16, kernel_size=(3, 3), stride=(1, 1))
  (conv3): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
  (conv4): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
  (dropout): Dropout2d(p=0.5, inplace=False)
  (fc1): Linear(in_features=1600, out_features = 512, bias=True)
  (quantum_layer): DressedQuantumNet(
    (pre_net): Linear(in_features=512, out_features=5, bias=True)
    (post_net): Linear(in_features=5, out_features=100, bias=True)
  )
)

## Reference Paper
- [Pennylane](https://pennylane.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Transfer learning in hybrid classical-quantum neural networks](https://arxiv.org/abs/1912.08278)
