# Introduction to TensorFlow Image Classification

## What makes TensorFlow a useful tool for learning ML?

- Graphs
- Sessions

Quick explanation of eager-execution and graph execution:
- eager-execution is what most programmers do to execute code, which is just immediate operation evaluations
- TensorFlow builds graphs of operations, and you use a ``Session`` object with passed input and output tensors, that would then compile it with ``session.run()``
