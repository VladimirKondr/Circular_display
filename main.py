import math
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def polar_to_xy(l, a):
    return l * math.cos(a), l * math.sin(a)


def xy_to_polar(x, y):
    return math.sqrt(x ** 2 + y ** 2), math.atan2(y, x)


def rgb_distance(color1, color2):
    return math.sqrt((color1[0] - color2[0]) ** 2 +
                     (color1[1] - color2[1]) ** 2 +
                     (color1[2] - color2[2]) ** 2)


def find_closest_color(target_color, color_list):
    if color_list is None:
        return target_color
    closest_color = None
    min_distance = float('inf')

    for color in color_list:
        distance = rgb_distance(target_color, color)
        if distance < min_distance:
            min_distance = distance
            closest_color = color

    return closest_color


class Dot:
    def __init__(self,
                 l: float = 0,
                 a: float = 0,
                 r: float = 0.58,
                 colors: list[str] = [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
                 brd: float = 0.12,
                 lum: float = 0,
                 color: str = np.full((3,), 1)):
        self.l = l
        self.a = a
        self.r = r
        self.colors = colors
        self.lum = lum
        self.brd = brd
        self.color = color
        self.acum_color_count = 0
        self.acum_color_sum = np.zeros((3,))

    def check(self):
        if self.lum == 0:
            raise Exception("Luminosity was set to 0 when firing")
        if self.lum < 0 or self.lum > 1:
            raise Exception("Luminosity is not in range (0, 1]")
        if self.colors is not None and self.color not in self.colors:
            raise Exception("There was an attempt to set dot to unaccessible color")

    def active(self):
        return self.lum != 0

    def fire(self, lum, color) -> bool:
        if self.active:
            return False
        self.lum = lum
        self.color = find_closest_color(color, self.colors)
        self.check()
        return True

    def unfire(self) -> bool:
        if not self.active:
            return False
        self.lum = 0
        self.color = None
        return True

    def set(self, lum, color) -> bool:
        self.lum = lum
        self.color = find_closest_color(color, self.colors)
        self.check()
        return True

    def size(self):
        return self.r + self.brd

    def acum(self, color):
        self.acum_color_count += 1
        self.acum_color_sum += color / 255
        return True

    def fire_acum(self):
        if self.acum_color_count == 0:
            if abs(self.a - (math.pi * 3/4)) < 0.1:
                print(self.l, self.a, math.pi * 3/4)
            return True
        avg_color = self.acum_color_sum / self.acum_color_count
        self.set(1, avg_color)
        self.acum_color_count = 0
        self.acum_color_sum = np.zeros((3,))
        return True

    def __str__(self):
        return f"Dot at ({self.l}, {self.a}) with luminosity {self.lum} and color {self.color} and acum {self.acum_color_sum}"


class Circle:
    def __init__(self,
                 r1: float = 1,
                 r: float = 0.58,
                 colors: list[str] = [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
                 brd: float = 0.12,
                 lum: float = 0,
                 color: str = np.full((3,), 1)):
        self.r1 = r1
        self.dots = []
        x = r + brd
        self.r2 = self.r1 + 2 * x
        self.count = int(math.pi / math.asin(x / (x + self.r1)))
        self.angle = 2 * math.pi / self.count
        for u in range(self.count):
            self.dots.append(Dot(r1 + x, self.angle * u, r, colors, brd, lum, color))

    def fire_all(self, lum, color) -> bool:
        for unit in self.dots:
            unit.fire(lum, color)
        return True

    def fire(self, a, lum, color) -> bool:
        a = a - (2 * math.pi) * (a // (2 * math.pi))
        if a < (self.angle / 2):
            return self.dots[0].fire(lum, color)
        a -= self.angle / 2
        x = a // self.angle
        if x + 1 == len(self.dots):
            x = -1
        return self.dots[1 + x].fire(lum, color)

    def acum(self, a, color):
        a = a - (2 * math.pi) * (a // (2 * math.pi))
        if a < (self.angle / 2):
            return self.dots[0].acum(color)
        a -= self.angle / 2
        x = a // self.angle
        if x + 1 == len(self.dots):
            x = -1
        return self.dots[int(1 + x)].acum(color)

    def fire_acum(self):
        for unit in self.dots:
            unit.fire_acum()

    def __str__(self):
        return f"Circle with radius {self.r1} and angle {self.angle} and count {self.count} and dots {self.dots}"


class Display:
    def __init__(self,
                 n: int = 0,
                 r: float = 0.58,
                 colors: list[str] = [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
                 brd: float = 0.12,
                 lum: float = 0,
                 color: str = np.full((3,), 1)):
        self.n = n
        self.center = Dot(0, 0, r, colors, brd, lum, color)
        self.circles = []
        x = r + brd
        for i in range(self.n):
            self.circles.append(Circle(x, r, colors, brd, lum, color))
            x += 2 * (r + brd)

    def bin_search(self, r):
        low = 0
        high = len(self.circles)
        while low <= high:
            mid = (low + high) // 2
            if mid == len(self.circles):
                return None
            mid_val = self.circles[mid]
            if mid_val.r1 <= r < mid_val.r2:
                return mid
            elif r < mid_val.r1:
                high = mid - 1
            else:
                low = mid + 1
        return None

    def fire(self, r, a, lum, color):
        if r < self.center.size():
            return self.center.fire(lum, color)
        x = self.bin_search(r)
        if x is None:
            raise Exception(f"No circle for Dot at ({r}, {a})")
        return self.circles[x].fire(a, lum, color)

    def size(self):
        return self.center.r if len(self.circles) == 0 else self.circles[-1].r2 - self.circles[-1].dots[0].brd

    def acum(self, l, a, color):
        if l < self.center.size():
            return self.center.acum(color)
        x = self.bin_search(l)
        if x is None:
            # raise Exception(f"No circle for Dot at ({l}, {a})")
            return False
        return self.circles[x].acum(a, color)

    def fire_acum(self):
        self.center.fire_acum()
        for unit in self.circles:
            unit.fire_acum()

    def __str__(self):
        x = f"Display with center {self.center} and circles\n"
        for unit in self.circles:
            x += unit.__str__() + "\n"
        return x


class Renderer:
    def __init__(self, display):
        self.display = display
        self.fig, self.ax = plt.subplots()

    def render_dot(self, dot):
        x, y = polar_to_xy(dot.l, dot.a)
        circle = plt.Circle((x, y), dot.r, color=dot.color if dot.lum > 0 else 'black', alpha=dot.lum)
        self.ax.add_patch(circle)

    def render_circle(self, circle):
        for dot in circle.dots:
            self.render_dot(dot)

    def render_display(self):
        self.render_dot(self.display.center)
        for circle in self.display.circles:
            self.render_circle(circle)

        max_radius = self.display.center.size() + max(circle.r2 for circle in self.display.circles)
        self.ax.set_xlim(-max_radius, max_radius)
        self.ax.set_ylim(-max_radius, max_radius)
        self.ax.set_aspect("equal")
        self.ax.axis('off')
        plt.show()

    def render_image(self, img: np.array):
        a1, b = len(img), len(img[0])
        for i in tqdm.tqdm(range(a1)):
            y = -i + a1 // 2
            for j in range(b):
                x = j - b // 2
                l, a = xy_to_polar(x, y)
                l = l * self.display.size() / (math.sqrt(a1 ** 2 + b ** 2) / 2)
                self.display.acum(l, a, img[i][j])
        self.display.fire_acum()


def main():
    img = PIL.Image.open("img_1.png")
    img = img.convert('RGB')
    img = np.array(img)
    x = Display(100, lum=1, brd=0.08, colors=None)
    ren = Renderer(x)
    ren.render_image(img)
    ren.render_display()


main()
