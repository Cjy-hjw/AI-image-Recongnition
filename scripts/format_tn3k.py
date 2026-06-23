from scripts.data.split_tn3k import format_split


def main():
    format_split(
        csv_path="data/TN3K/label4trainval.csv",
        img_dir="data/TN3K/trainval-image",
        mask_dir="data/TN3K/trainval-mask",
        dest_dir="data/train",
        split_ratio=0.85,
        is_trainval=True,
    )
    format_split(
        csv_path="data/TN3K/label4test.csv",
        img_dir="data/TN3K/test-image",
        mask_dir="data/TN3K/test-mask",
        dest_dir="data/holdout",
        is_trainval=False,
    )


if __name__ == "__main__":
    main()
