import os
import yaml


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            fp = cfg.get("defaults").get(d)
            cf = os.path.join(os.path.dirname(config_file), fp)
            with open(cf) as f:
                val = yaml.safe_load(f)
                print(val)
                cfg.update(val)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg
