"""attacker module for langsec"""

def main(args=None) -> None:

    import argparse

    parser = argparse.ArgumentParser(
        prog="LangSec",
        description="A suit to perform attacks against Language Models and generate a report",
        epilog="See https://github.com/l3th4l/LangSec"
    )

    parser.add_argument(
        "--parallel_requests",
        type=int,
        default=6, # TODO to be change later and replace this with a config file
        help="How many generator requests to launch in parallel for a given prompt. Ignored for models that support multiple generations per call.",
    )
    parser.add_argument(
        "--parallel_attempts",
        type=int,
        default=6, # TODO to be change later and replace this with a config file
        help="How many probe attempts to launch in parallel. Raise this for faster runs when using non-local models.",
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=1337, # TODO to be change later and replace this with a config file
        help="random seed",
    )

    parser.add_argument(
        "--model_type",
        "-m",
        type=str,
        help="type of the target model, e.g. 'huggingface', 'openai', 'azure', 'anthropic', 'ollama'",
    )
    parser.add_argument(
        "--model_name",
        "-n",
        type=str,
        default=None,
        help="name of the target model, e.g. 'gpt-4', 'gpt-3.5-turbo', 'claude-2', 'Llama-2-70b-chat', etc.",
    )

    # TODO implement subcommands for different attacks

    parser.print_help()

