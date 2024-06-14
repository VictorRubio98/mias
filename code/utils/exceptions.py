

class ModelNotFit(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class NotAvailableFeature(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
