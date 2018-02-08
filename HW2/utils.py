import numpy as np
import torchtext 
import torch

def validate(model, val_iter):
	correct = 0.0
	total  = 0.0
	for batch in val_iter:
		x = batch.text
		probs = model(x)
		print(probs)
		_, preds = torch.max(probs, 1)
		print(preds)

		# cut off the first target for now
		targets = x[0, 1:]
		print(x)
		print(targets)
		# cut off the last prediction for now
		preds = preds[:-1]

		correct += sum(preds == targets.data).data.np()[0]
		total += batch.text.size()[0] - 1
	return correct / total
