import json
from argparse import ArgumentParser

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from bottle_vision.dataset import IllustDataset
from bottle_vision.model import IllustEmbeddingModel

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def collate_fn(batch):
    # Custom collate function to handle dataclass objects
    if len(batch) == 0:
        return batch
    if batch[0] is None:
        return None
    if hasattr(batch[0], "__dataclass_fields__"):
        return type(batch[0])(*[collate_fn([getattr(d, f) for d in batch]) for f in batch[0].__dataclass_fields__])
    elif isinstance(batch[0], (tuple, list)):
        return type(batch[0])(collate_fn(samples) for samples in zip(*batch))
    else:
        return torch.utils.data._utils.collate.default_collate(batch)


def extract_embed(src_ckpt_path: str, dest_emb_path: str, config_path: str, artist_indices_path: str):
    # config yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # artist indices
    with open(artist_indices_path) as f:
        artist_indices = json.load(f)
    print(f"Loaded artist indices from {artist_indices_path}, {len(artist_indices)} artists")
    artist_batches = list(artist_indices.values())

    # load checkpoint
    ckpt = torch.load(src_ckpt_path)
    state_dict = ckpt["state_dict"]
    model_hparams = ckpt["hyper_parameters"]
    print(f"Loaded checkpoint from {src_ckpt_path}")
    print(model_hparams)

    # model
    model = IllustEmbeddingModel(
        num_tags=model_hparams["num_tags"],
        num_artists=model_hparams["num_artists"],
        num_characters=model_hparams["num_characters"],
        backbone_variant="vit_base_patch16_224",
        image_size=model_hparams["image_size"],
        tag_embed_dim=model_hparams["tag_embed_dim"],
        artist_embed_dim=model_hparams["artist_embed_dim"],
        character_embed_dim=model_hparams["character_embed_dim"],
        cls_token=model_hparams["cls_token"],
        reg_tokens=model_hparams["reg_tokens"],
        dropout=model_hparams["dropout"],
        tag_temp=model_hparams["tag_contrastive_config"]["temp"],
        artist_temp=model_hparams["artist_contrastive_config"]["temp"],
        character_temp=model_hparams["character_contrastive_config"]["temp"],
        tasks=model_hparams["tasks"],
        temp_strategy=model_hparams["temp_strategy"],
    )
    model = torch.compile(model)
    torch.set_float32_matmul_precision("high")

    # load state dict
    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"Loaded model from {src_ckpt_path}")

    # data
    dataset = IllustDataset(
        parquet_path=config["data"]["train_parquet_path"],
        tar_dir=config["data"]["train_tar_dir"],
        num_tags=config["data"]["num_tags"],
        num_artists=config["data"]["num_artists"],
        num_characters=config["data"]["num_characters"],
        tasks=[],
        image_size=config["data"]["image_size"],
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=artist_batches,
        num_workers=16,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # inference, collect artist embeddings
    artist_embeds = []
    model.eval()
    model = model.to(DEVICE)
    for batch in tqdm(dataloader, desc="Inference"):
        with torch.no_grad():
            features = model.backbone(batch.image.to(DEVICE))
            artist_emb = model.artist_head(features)

        # average over samples
        artist_emb = artist_emb.mean(dim=0)
        artist_embeds.append(artist_emb)
    print(f"Collected {len(artist_embeds)} artist embeddings, {artist_embeds[0].shape}")

    artist_embeds = torch.stack(artist_embeds)
    torch.save(artist_embeds, dest_emb_path)


def update_weight(src_ckpt_path: str, dest_ckpt_path: str, embed_path: str):
    state_dict = torch.load(src_ckpt_path)["state_dict"]
    artist_prototypes = state_dict["model._orig_mod.artist_prototypes"]
    print(f"Loaded artist prototypes from {src_ckpt_path}, {artist_prototypes.shape}")

    artist_embeds = torch.load(embed_path)
    print(f"Loaded artist embeddings from {embed_path}, {artist_embeds.shape}")

    artist_prototypes = torch.concat([artist_prototypes, artist_embeds], dim=0)
    print(f"Updated artist prototypes, {artist_prototypes.shape}")

    state_dict["model._orig_mod.artist_prototypes"] = artist_prototypes
    ckpt = {"state_dict": state_dict}
    torch.save(ckpt, dest_ckpt_path)
    print(f"Saved updated model to {dest_ckpt_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument("src_ckpt_path", type=str)
    parser.add_argument("dest_ckpt_path", type=str)
    parser.add_argument("dest_emb_path", type=str)
    parser.add_argument("config_path", type=str)
    parser.add_argument("artist_indices_path", type=str)
    args = parser.parse_args()

    extract_embed(args.src_ckpt_path, args.dest_emb_path, args.config_path, args.artist_indices_path)
    update_weight(args.src_ckpt_path, args.dest_ckpt_path, args.dest_emb_path)


if __name__ == "__main__":
    main()
