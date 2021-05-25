class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.distance = 0

    def print(self):
        print("x:", self.x, " y:", self.y)
        pass

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def print_p(self):
        print("x:",self.x," y:",self.y)