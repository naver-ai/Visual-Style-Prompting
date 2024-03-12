
from PIL import Image

def get_image(image_path, row, col, image_size=1024, grid_width=1):

    left_point = (image_size + grid_width) * col
    up_point = (image_size + grid_width) * row
    right_point = left_point + image_size
    down_point = up_point + image_size

    if type(image_path) is str:
        image = Image.open(image_path)
    else:    
        image = image_path
    croped_image = image.crop((left_point, up_point, right_point, down_point))
    return croped_image

def get_image_v2(image_path, row, col, image_size=1024, grid_row_space=1, grid_col_space=1):

    left_point = (image_size + grid_col_space) * col
    up_point = (image_size + grid_row_space) * row
    right_point = left_point + image_size
    down_point = up_point + image_size

    if type(image_path) is str:
        image = Image.open(image_path)
    else:    
        image = image_path
    croped_image = image.crop((left_point, up_point, right_point, down_point))
    return croped_image

def create_image(row, col, image_size=1024, grid_width=1, background_color=(255,255,255), top_padding = 0, bottom_padding = 0, left_padding = 0, right_padding = 0):

    image = Image.new('RGB', (image_size * col + grid_width * (col - 1) + left_padding , image_size * row + grid_width * (row - 1)), background_color)
    return image

def paste_image(grid, image, row, col, image_size=1024, grid_width=1, top_padding = 0, bottom_padding = 0, left_padding = 0, right_padding = 0):
    left_point = (image_size + grid_width) * col + left_padding
    up_point = (image_size + grid_width) * row + top_padding
    right_point = left_point + image_size
    down_point = up_point + image_size
    grid.paste(image, (left_point, up_point, right_point, down_point))

    return grid

def paste_image_v2(grid, image, row, col, grid_size=1024, grid_width=1, top_padding = 0, bottom_padding = 0, left_padding = 0, right_padding = 0):
    left_point = (grid_size + grid_width) * col + left_padding
    up_point = (grid_size + grid_width) * row + top_padding
    
    image_width, image_height = image.size

    right_point = left_point + image_width 
    down_point = up_point + image_height

    grid.paste(image, (left_point, up_point, right_point, down_point))

    return grid


def pivot_figure(file_path, image_size=1024, grid_width=1):
    if type(file_path) is str:
        image = Image.open(file_path)
    else:
        image = file_path
    image_col = image.width // image_size
    image_row = image.height // image_size


    grid = create_image(image_col, image_row, image_size, grid_width)

    for row in range(image_row):
        for col in range(image_col):
            croped_image = get_image(image, row, col, image_size, grid_width)
            grid = paste_image(grid, croped_image, col, row, image_size, grid_width)
    
    return grid

def horizontal_flip_figure(file_path, image_size=1024, grid_width=1):
    if type(file_path) is str:
        image = Image.open(file_path)
    else:
        image = file_path
    image_col = image.width // image_size
    image_row = image.height // image_size

    grid = create_image(image_row, image_col, image_size, grid_width)

    for row in range(image_row):
        for col in range(image_col):
            croped_image = get_image(image, row, image_col - col - 1, image_size, grid_width)
            grid = paste_image(grid, croped_image, row, col, image_size, grid_width)

    return grid

def vertical_flip_figure(file_path, image_size=1024, grid_width=1):
    if type(file_path) is str:
        image = Image.open(file_path)
    else:
        image = file_path

    image_col = image.width // image_size
    image_row = image.height // image_size

    grid = create_image(image_row, image_col, image_size, grid_width)

    for row in range(image_row):
        for col in range(image_col):
            croped_image = get_image(image, image_row - row - 1, col, image_size, grid_width)
            grid = paste_image(grid, croped_image, row, col, image_size, grid_width)

    return grid