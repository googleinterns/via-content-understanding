class ExpertStore:

    keymap = None
    keys = None
    store = None

    def todict(self):
        return {key: self.store[index] for (key, index) in self.keymap.items()}