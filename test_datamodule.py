import logging
from bottle_vision.datamodule import IllustDataModule


def test():
    data_module = IllustDataModule(
        train_parquet_path="../danbooru2023/export/train/images.parquet",
        train_tar_dir="../danbooru2023/export/train",
        train_tag_dict_path="../danbooru2023/export/train/tag_indices.json",
        train_artist_dict_path="../danbooru2023/export/train/artist_indices.json",
        train_character_dict_path="../danbooru2023/export/train/character_indices.json",
        valid_parquet_path="../danbooru2023/export/valid/images.parquet",
        valid_tar_dir="../danbooru2023/export/valid",
        classes_per_batch=4,
        samples_per_class=2,
        num_tags=8110,
        num_artists=1024,
        num_characters=2751,
        label_smoothing_eps=0.1,
        num_workers=1,
    )

    data_module.setup("train")
    
    print("setting train dataloader")
    train_dataloader = data_module.train_dataloader()
    iter(train_dataloader)
    
    print("setting valid dataloader")
    val_dataloader = data_module.val_dataloader()
    iter(val_dataloader)

    for i, batch in enumerate(train_dataloader):
        if i >= 3:
            break
        print(batch)

    for i, batch in enumerate(val_dataloader):
        if i >= 3:
            break
        print(batch)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test()
