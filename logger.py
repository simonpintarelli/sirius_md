class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    """Logger for timestepping scheme."""
    def __init__(self):
        self.active = False
        self.current = {}
        self.log = []

    def __enter__(self):
        self.log = []
        self.active = True

    def __exit__(self, cls, value, traceback):
        self.active = False
        if len(self.current) > 0:
            self.log.append(self.current)
            self.current = {}

    def insert(self, entries):
        """insert entries (dict)"""
        if not self.active:
            return

        if set(entries) < set(self.current):
            #assume next round
            self.log.append(self.current)
            self.current = {}

        self.current = {**self.current, **entries}
