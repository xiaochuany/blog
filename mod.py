class ExampleClass:
    """
    A simple example class to demonstrate Google style docstrings.

    """

    def __init__(self, a:str, b: int):
        """
        Initializes ExampleClass with param1 and param2.

        Args:
            a : The first parameter.
            b : The second parameter.
        """
        self.param1 = a
        self.param2 = b
 
    def method1(self, arg:str):
        """
        Example method1 oo.

        Parameters:
            arg: a string argument to be printed.

        Returns:
            str: the same string that was passed as an argument.
        """
        print(arg)
        return arg

    def method2(self):
        """
        Example method2 that prints param2.

        Returns:
            None
        """
        print(self.param2)


def calculate_area(length: float, 
                   width: float):
    """
    Calculates the area of a rectangle.

    Args:
        length: The length of the rectangle.
        width: The width of the rectangle.

    Returns:
        float: The area of the rectangle.

    Raises:
        ValueError: If length or width is negative.
    """
    if length < 0 or width < 0:
        raise ValueError("Length and width must be non-negative.")
    return length * width