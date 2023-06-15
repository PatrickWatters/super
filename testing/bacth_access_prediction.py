#code to try simulate the batch access time predictions


TRAINING_SPEED = 0.1
TOTAL_DELAY = 0
NUM_BTACHES = 196
batches_remianing  = list(range(1, NUM_BTACHES+1))

actual_access_times  = {}
predicted_access_times  = {}

for idx in range(0, len(batches_remianing)):
    batch_id= batches_remianing[idx]
    actual_access_times[batch_id] = idx * TRAINING_SPEED

for idx in range(0, len(batches_remianing)):
    batch_id= batches_remianing[idx]
    predicted_access_times[batch_id] = idx * TRAINING_SPEED
    
assert predicted_access_times == actual_access_times




