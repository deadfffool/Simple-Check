from ADTopklib import Memory


class NoneMemory(Memory):
    
    
    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass

    def update_gtopk(self,ctx,tensor_decompressed, tensors_aggregated_avg, original_tensor, name):
        """Update the residuals."""
        pass
    
    def compensate_gtopk(self, tensors_aggregated_avg, name):
        """Update the tensor with the residuals."""
        pass
