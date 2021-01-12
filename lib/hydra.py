import argparse


def get_config_name():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name")
    args, unknown = parser.parse_known_args()
    return args.config_name
