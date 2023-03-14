from enum import EnumMeta, Enum


class EnumMetaContains(EnumMeta):
    def __contains__(cls, value):
        if isinstance(value, cls):
            return True
        return False


class EnumContains(Enum, metaclass=EnumMetaContains):
    pass


#  The following enum is not used by the communication library.
#  Is up to the user to implement what the node does upon receiving
#    one of the messages. And when to send them in place of the actual values.
#  Don't use it if you don't need it.
class Signals(EnumContains):
    MISSING_VALUE = 1  # Use it in read/write_format as a default value
    NOT_OBSERVED = 2  # Signal that the value is missing because it wasn't observed
    USE_LATEST = 3  # Signal the receiver to use the latest valid value