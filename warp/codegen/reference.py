class Reference:
    def __init__(self, value_type):
        self.value_type = value_type


def is_reference(type):
    return isinstance(type, Reference)


def strip_reference(arg):
    if is_reference(arg):
        return arg.value_type
    else:
        return arg
