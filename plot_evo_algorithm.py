import evolutionary
import show_surface
import numpy as np
import string
import random


def random_string(size):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(size))


def value_to_color(value):
    # Calculate the red and green values
    red = int((1 - value) * 255)
    green = int(value * 255)
    
    # Construct the color code
    color_code = "#{:02x}{:02x}{:02x}".format(red, green, 0)
    
    return color_code


def plot_evolutionary_algorithm(fitness_function, initial_population, mutation_rate, mutation_stddev, n_steps, every_n=None):
    population = initial_population
    pop_averages = []
    pop_states = []
    for i in range(n_steps):
        population = evolutionary.evolutionary_step(population, fitness_function, mutation_rate, mutation_stddev)
        
        if (every_n is None) or (i % every_n == 0) or (i == n_steps-1):
            pop_average = np.zeros_like(population[0])
            for member in population:
                pop_average += member
            
            pop_average /= len(population)
            print([*pop_average, fitness_function(pop_average)])
            
            pop_averages.append(pop_average)
            pop_states.append(population)
    
    min_vals = np.amin(pop_averages, axis=0)
    max_vals = np.amax(pop_averages, axis=0)

    min_x = min_vals[0]
    min_y = min_vals[1]
    max_x = max_vals[0]
    max_y = max_vals[1]
    
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    fig_list = []
    for i, point in enumerate(pop_averages):
        
        displayed_points = [point]
        displayed_colors = ['#000000']
        zorders = [2]
        
        for pop in pop_states[i]:
            displayed_points.append(pop)
            displayed_colors.append('#FFFFFF')
            zorders.append(1)
        
        fig = show_surface.plot_mesh_with_point(
            fitness_function, (min_x, min_y), (x_range, y_range), 20, 'cool', displayed_points, displayed_colors, zorders)
        fig_list.append(fig)
    image_list = show_surface.get_image_list(fig_list)
    anim = show_surface.create_animation_from_images(image_list, interval=50)
    anim_id = random_string(4)
    anim.save('evo_animation_{}_{}_{}.mp4'.format(mutation_rate, mutation_stddev, anim_id), fps=2, extra_args=['-vcodec', 'libx264'])
    

def sinxy_(x, y):
    return (np.sin(np.sqrt(x ** 2 + y ** 2)) + 1) / 2


def sinxy(point1, point2=None):
    if point2 is None:
        return sinxy_(point1[0], point1[1])
    else:
        return sinxy_(point1, point2)


if __name__ == "__main__":
    starting_point = np.array([3.4, 3.4], dtype=float)
    initial_pop = evolutionary.generate_normdist_points(starting_point, 500, 0.2)

    plot_evolutionary_algorithm(sinxy, initial_pop, 0.2, 0.015, 2000, 100)
