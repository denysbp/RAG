class CliError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


class ModelsError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)
