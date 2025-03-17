import os
from collections import defaultdict

import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir: str):
        super().__init__(write_interval="epoch")
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # collect predictions
        filename_list = []
        output_dict = defaultdict(list)
        for output, filenames in predictions:
            for key, value in output.__dict__.items():
                if isinstance(value, torch.Tensor):
                    output_dict[key].append(value)
            filename_list.extend(filenames)

        # stack tensors
        for key, value in output_dict.items():
            output_dict[key] = torch.cat(value, dim=0)

        # save predictions
        output_dict["filename"] = filename_list
        output_dict = {key: value for key, value in output_dict.items()}
        dest_path = os.path.join(self.output_dir, "prediction.pt")
        torch.save(output_dict, dest_path)
