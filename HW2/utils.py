import numpy as np
import torchtext 
import torch

def validate(model, val_iter):
	correct = 0.0
	total  = 0.0
	last_pred = None
	for batch in val_iter:
		x = batch.text
		probs = model(x)
		_, preds = torch.max(probs, 1)

		# cut off the first target unless there is a previous prediction
		targets = x[0, :] if last_pred is not None else x[0, 1:]
		# cut off the last prediction for now
		current_preds = preds[:-1]
		current_preds = torch.cat([last_pred, current_preds]) if last_pred is not None else current_preds

		last_pred = preds[-1:]

		correct += sum(current_preds == targets.data)
		total += batch.text.size()[1] - 1
	return correct / total
