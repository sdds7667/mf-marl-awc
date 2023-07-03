import argparse
from pathlib import Path
from typing import Optional
import sys

from utils.runner import Runner

sys.path.append(str(Path(__file__).parent))

from utils.run_config import RunConfig

run_config: Optional[RunConfig] = None


def program_arguments() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(description='Wind Farm Experiment')
    # Optional argument
    arg_parser.add_argument("run_name", type=str, help="name of the run", default=None, nargs="?")
    return arg_parser.parse_args()


def find_config_file(configuration_file_string: str, confirm=False) -> Optional[RunConfig]:
    """Find the configuration file with the given name.

    The file name can be partial. If you want to specify which file, you can specify the parent folders partially as well.
    Example: "3x1/basic" will find all files basic. If there are more, it will look at the parent folder which matches
    "3x1" and so on.

    :param configuration_file_string: The name of the configuration file to find.
    :param confirm: Whether to confirm the file to use.
    :return: The RunConfiguration loaded from that file.
    """

    path_filter = ["*" + x + "*" for x in configuration_file_string.split("/")]
    path_filter[-1] += ".py"
    path_filter = "/".join(path_filter)

    files = (list(Path("runs").glob("**/" + path_filter)))
    if len(files) == 0:
        print("No files found")
        return None
    elif len(files) > 1:
        print("Multiple files found:")
        for i, file in enumerate(files):
            print(f"{i}: {str(file.absolute())}")
        return None
    else:
        if confirm:
            input(f"Found file: \"{files[0]}\" [CTRL+C] to cancel, Enter to continue")

        import importlib.util
        spec = importlib.util.spec_from_file_location("run_config", files[0])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config


if __name__ == "__main__":
    if run_config is None:
        # Try the program arguments
        args = program_arguments()
        if args.run_name is not None:
            run_config = find_config_file(args.run_name)
        else:
            print("Handles partial names. For example: 3x1/basic will match \"*3x1*/*basic*.py\"")
            config_name = input("Config Name > ")
            run_config = find_config_file(config_name, confirm=True)

    if run_config is None:
        raise ValueError("No run config found")
    Runner(run_config).optimize()
