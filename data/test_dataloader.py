from data_loader import Loaders

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    config = parser.parse_args()

    config.image_size = [384, 128]
    config.dataset_path = "/home/Monday/datasets/SYSU/"
    config.p = 16
    config.k = 8
    loaders = Loaders(config)

    trainloader = loaders.rgb_ir_train_loader
