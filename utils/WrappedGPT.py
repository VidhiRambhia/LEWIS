import torch.nn as nn 
import torch

class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        # Initialize the layer, device, rows, and columns
        self.layer = layer
        self.layer_id = layer_id 
        self.layer_name = layer_name if layer_name != "none" else layer.__class__.__name__
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        # Initialize the scaler row and number of samples basically used for normalizing
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0
        self.activations_list = []

    def add_batch(self, inp, out):
        """
        This method is used to update the scaler row of the WrappedGPT class.
        The scaler row is used for normalizing the input data. It is necessary 
        to update the scaler row for each batch of input data to ensure that the 
        normalization is accurate.

        Args:
            inp (torch.tensor): The input data.
            out (torch.tensor): The output data.

        Returns:
            None
        """
        # If input is 2D, add a batch dimension
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        
        # Store the batch size
        tmp = inp.shape[0]
        
        # If the layer is a Linear layer, reshape the input
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp
        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        self.activations_list.append(inp)