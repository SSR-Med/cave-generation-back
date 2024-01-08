# Import libraries
# SSR
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
import pygad
from PIL import Image, ImageDraw, ImageFont


def create_points(widht: int, height: int, rooms: int):
    x_points = [random.uniform(0, widht) for x in range(widht//10)]
    y_points = [random.uniform(0, height) for y in range(height//10)]
    classifier = KMeans(n_clusters=rooms, n_init="auto").fit(
        [[x_points[i], y_points[i]] for i in range(len(x_points))])
    return classifier.cluster_centers_

# Normal distribution with box muller


def normal(avg: int, dev: int):
    ans = -1
    while ans < 1:
        r1 = random.random()
        r2 = random.random()
        x1 = math.sqrt(-2*math.log(r1))*math.cos(2*math.pi*r2)
        ans = avg+(x1*dev)
    ans = int(ans)
    return ans if ans % 2 == 0 else ans-1


def create_rooms(widht: int, height: int, center_rooms: list):
    # Rules:
    # 1. Rooms will be generated with a normal function
    # 2. Each room should be inside the widht and height (xRight:0,yTop:0,xLeft:0,yBottom:0)
    # 3. Rooms will generate as a list of tuples [(xRight:0,yTop:0,xLeft:0,yBottom:0)]

    # Iterate rooms
    rooms = []
    for c_room in center_rooms:
        height_sum = normal(height//10, height//50)//2
        width_sum = normal(widht//10, widht//50)//2
        # Create coordinates tuple
        room_coordinates = [c_room[0], c_room[1], c_room[0], c_room[1]]
        for i in range(4):
            # First: Sum to the dimensions
            # Check if it is X or Y
            if i % 2 == 0:
                sum = width_sum
            else:
                sum = height_sum
            # Check if it is Right/top or Left/Down
            if i < 2:
                room_coordinates[i] = room_coordinates[i] + sum
                # Second: Check if the coordinate is between the limits, if not sum it to the other edge
                if room_coordinates[i] > widht:
                    room_coordinates[i] = room_coordinates[i] - sum
                    room_coordinates[i+2] = room_coordinates[i+2] - sum
            else:
                room_coordinates[i] = room_coordinates[i] - sum
                # Second: Check if the coordinate is between the limits, if not sum it to the other edge
                if room_coordinates[i] < 0:
                    room_coordinates[i] = room_coordinates[i] + sum
                    room_coordinates[i-2] = room_coordinates[i-2] + sum
        rooms.append([int(x) for x in room_coordinates])
    return rooms


def distance_centers(centers: list):
    distance_matrix = []
    for j in range(len(centers)):
        distances = []
        for i in range(len(centers)):
            if j == i:
                distance = 0
            else:
                distance = (((centers[i][0]-centers[j][0])**2) +
                            ((centers[i][1]-centers[j][1])**2))**0.5
            distances.append(distance)
        distance_matrix.append(distances)
    return distance_matrix


def nearest_rooms(distance_matrix: list, relations: dict):
    for j in range(len(distance_matrix)):
        minDistance = 10000000000000000000
        min_i = 30
        for i in range(len(distance_matrix)):
            if i != j and distance_matrix[j][i] < minDistance and i not in relations[j]:
                minDistance = distance_matrix[j][i]
                min_i = i
        relations[min_i].add(j)
        relations[j].add(min_i)


def traveler(distance_matrix: list, relations: dict, num_rooms: int):
    # 1. Create fitness function
    def fitness_function(ga_instance, solution, solution_idx):
        fitness = 0
        for i in range(num_rooms-1):
            j = i+1
            fitness -= distance_matrix[int(solution[i])][int(solution[j])]
        return fitness
    # 2. Parameters for genetic algorithm
    num_generations = 300
    population_size = 50
    mutation_percent_genes = 10
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=population_size//2,
                           fitness_func=fitness_function,
                           sol_per_pop=population_size,
                           num_genes=num_rooms,
                           gene_space=list(range(num_rooms)),
                           mutation_percent_genes=mutation_percent_genes,
                           parent_selection_type="rank",
                           crossover_type="single_point",
                           mutation_type="random",
                           keep_parents=-1,
                           gene_type=int,
                           allow_duplicate_genes=False)
    ga_instance.run()
    solution = [int(x) for x in ga_instance.best_solution()[0]]
    for i in range(num_rooms-1):
        j = i+1
        relations[solution[i]].add(solution[j])
        relations[solution[j]].add(solution[i])


def draw_map(width, height, map):
    '''
    img = Image.new(mode="RGB", size=(width, height), color=(0, 0, 0))
    for row in range(height+1):
        for column in range(width+1):
            if map[row][column] == "*":
                img.putpixel((column, row), (255, 255, 255))
                # img[row][column] = (40, 42, 255)
            elif map[row][column] == "-":
                img.putpixel((column, row), (61, 183, 228))
    img.save("static/images/generation.jpg")
    '''
    char_width = width//10
    char_height = height//10
    width = char_width * width
    height = char_height * height
    image = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    #font = ImageFont.truetype("arial.ttf", size=char_height)
    font = ImageFont.truetype("static/FreeMono.ttf", encoding="unic",size=char_height)
    for i, row in enumerate(map):
        for j, char in enumerate(row):
            x = j * char_width
            y = i * char_height
            if char == "*" or char == "-" or char == "|":
                draw.text((x, y), char, font=font, fill=(255, 255, 255))
            else:
                draw.text((x, y), char, font=font, fill=(0, 0, 0))
    image.save("static/generation.jpg")


def create_dungeon(size: str):
    # Check what size the user wants
    if size == "big":
        width = 500
        height = 500
    else:
        width = 250
        height = 250
    # How many rooms?
    numRooms = width//10
    # Create center rooms [[xi,yi]]
    center_rooms = create_points(width, height, numRooms)
    # Time to create the rooms
    rooms = create_rooms(width, height, center_rooms)
    # We are going to create the paths between rooms
    # 1. Nearest room connection
    # 2. Traveler problem connection
    relations = {x: set() for x in range(numRooms)}
    # Create distances matrix
    distance_matrix = distance_centers(center_rooms)
    # 1. Nearest room connection
    nearest_rooms(distance_matrix, relations)
    # 2. Traveler problem connection
    # Character starts top left, end bottom right
    traveler(distance_matrix, relations, numRooms)
    # Finally: The creation of the dungeon
    # 1. Create map width x height full of walls
    map = [["#" for x in range(width+1)] for y in range(height+1)]
    # 2. Create the rooms
    # Iterate
    for room in rooms:
        # (xRight:0,yTop:0,xLeft:0,yBottom:0)
        for row in range(room[3], room[1]+1):
            for column in range(room[2], room[0]+1):
                map[row][column] = "*"
    # 3. Create the relationships between the rooms
    center_rooms = [[int(x) for x in center] for center in center_rooms]
    for index_room in range(numRooms):
        if len(relations[index_room]) != 0:
            for connection in relations[index_room]:
                # X,Y current element
                current_center = center_rooms[index_room]
                # X,Y other element
                other_center = center_rooms[connection]
                # Choose the element that is in the left
                if current_center[0] < other_center[0]:
                    left_center = current_center
                    right_center = other_center
                else:
                    left_center = other_center
                    right_center = current_center
                # Time to create the relations
                # 1. Left to right
                for i in range(left_center[0], right_center[0]+1):
                    map[left_center[1]][i] = "*"
                # 2. Vertical relations
                if right_center[1] > left_center[1]:
                    vertical_range = range(left_center[1], right_center[1])
                else:
                    vertical_range = range(right_center[1], left_center[1])
                # Create it
                for i in vertical_range:
                    map[i][right_center[0]] = "*"
                # Remove the element from the other set
                relations[connection].remove(index_room)
    # Create the walls (-)
    for row in range(height+1):
        for column in range(width+1):
            # Check if it is floor
            if map[row][column] == "*":
                # Up, UpRight, Right, DownRight, Down, DownLeft, Left, UpLeft
                # movements = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
                movements = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                for movement in movements:
                    # Check if it is inside the limits of the map
                    if 0 <= column+movement[1] <= 1000 and 0 <= row+movement[0] <= 1000:
                        # If it is left/right -> |, else -
                        if movement == (0, 1) or movement == (0, -1):
                            changeChar = "|"
                        else:
                            changeChar = "-"
                        # If it is in the limit -> Wall
                        if column+movement[1] in (1000, 0) or row+movement[0] in (1000, 0):
                            map[row+movement[0]][column +
                                                 movement[1]] = changeChar
                        elif map[row+movement[0]][column+movement[1]] == "#":
                            map[row+movement[0]][column +
                                                 movement[1]] = changeChar  # If empty

    draw_map(width, height, map)
    map = "\n".join(["".join(element) for element in map])
    return map
