import configparser

def read_config(file_path) -> configparser.ConfigParser:
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(file_path)
    return config

