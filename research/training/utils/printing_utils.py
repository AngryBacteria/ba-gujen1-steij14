def print_with_heading(to_print=""):
    if to_print == "":
        print(f"{60 * '='}")
    else:
        print(f"{30 * '='} {to_print} {30 * '='}")


def print_welcome_message():
    print(
        r"""
 ____       _              _____   _                    _       
| __ )     / \            |_   _| | |__     ___   ___  (_)  ___ 
|  _ \    / _ \    _____    | |   | '_ \   / _ \ / __| | | / __|
| |_) |  / ___ \  |_____|   | |   | | | | |  __/ \__ \ | | \__ \
|____/  /_/   \_\           |_|   |_| |_|  \___| |___/ |_| |___/
"""
    )