# change the behaviour of a class as a `type`
class MetaFoo(type):
    # https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python
    # def __new__(cls, clsname, bases, dct):
    #     pass

    def __getitem__(self, Base):
        r"""Intuitive syntactic sugar for class wrapping."""
        class cls(Base, Foo):
            # custom code
            pass

        # update defaults
        name = f"Foo{Base.__name__}"
        setattr(cls, "__name__", name)
        setattr(cls, "__qualname__", name)
        # mimicing '__module__' is not a good idea, so we keep '__doc__' only
        if hasattr(Base, "__doc__"):
            setattr(cls, "__doc__", getattr(Base, "__doc__"))

        return cls


class Foo(metaclass=MetaFoo):
    def __repr__(self):
        return f"{type(self)}()"
