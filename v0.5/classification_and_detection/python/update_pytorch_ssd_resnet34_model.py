import torch
from models.ssd_r34 import SSD_R34

# Load the reference model + weights.
model = torch.load('/datasets/mlperf-v0.5/models/resnet34-ssd1200.pytorch', map_location=lambda storage, loc: storage)
# Save just the MLPerf *weights*.
torch.save(model.state_dict(), 'resnet34-ssd1200.state_dict.pytorch')

# Create a uninitialized model using the newer PyTorch API to support quantization and JIT:
updated_model = SSD_R34()
# Overlay the saved MLPerf *weights* onto the updated model definition:
updated_model.load_state_dict(torch.load('resnet34-ssd1200.state_dict.pytorch'))
# Save the updated model (updated definition + MLPerf weights):
torch.save(updated_model, 'resnet34-ssd1200.updated.pytorch')
