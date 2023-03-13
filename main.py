import evolutionary
import show_surface
import numpy as np
import string
import random
import matplotlib
matplotlib.rcParams['figure.max_open_warning'] = 0


def sinxy_derivative(x, y):
    sqrt_xy = np.sqrt(x**2 + y**2)
    bottom = 2 * sqrt_xy
    return [ (x * np.cos(sqrt_xy)) / bottom, (y * np.cos(sqrt_xy)) / bottom ]


def random_string(size):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(size))


def calculate_pop_average(population):
    pop_average = np.zeros_like(population[0])
    for member in population:
        pop_average += member
    
    pop_average /= len(population)
    
    return pop_average


def get_abs_diff(value1, value2):
    return np.sum(np.abs(value1 - value2))


def normalized_dot(arr1, arr2):
    return np.dot(arr1 / np.linalg.norm(arr1), arr2 / np.linalg.norm(arr2))


def plot_evolutionary_algorithm(fitness_function, initial_population, mutation_rate, mutation_stddev, n_steps, every_n=None):
    population = initial_population
    pop_averages = []
    pop_states = []
    pop_delta_total = np.zeros_like(population[0])
    pop_average = calculate_pop_average(population)
    
    for i in range(n_steps):
        
        next_population = evolutionary.evolutionary_step(population, fitness_function, mutation_rate, mutation_stddev)
        
        pop_average = calculate_pop_average(population)
        next_pop_average = calculate_pop_average(next_population)
        
        pop_delta_total += next_pop_average - pop_average
        
        if (every_n is None) or (i % every_n == 0) or (i == n_steps-1):
            
            print([*next_pop_average, fitness_function(next_pop_average)])
            print("______________________________________")
            
            pop_averages.append(next_pop_average)
            pop_states.append(next_population)
        
        # update the population
        population = next_population
    
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
            fitness_function,
            (min_x, min_y),
            (x_range, y_range),
            20,
            'cool',
            displayed_points,
            displayed_colors,
            zorders
        )
        fig_list.append(fig)
    anim_id = random_string(4)
    print("Animation id: ", anim_id)
    show_surface.create_and_save_anim(fig_list, 'animations/evo_animation_{}_{}_{}.mp4'.format(mutation_rate, mutation_stddev, anim_id))


def sinxy_(x, y):
    return (np.sin(np.sqrt(x ** 2 + y ** 2)) + 1) / 2


def sinxy(point1, point2=None):
    if point2 is None:
        return sinxy_(point1[0], point1[1])
    else:
        return sinxy_(point1, point2)


def test_evo_algorithm():
    starting_point = np.array(np.random.random(2) * 6, dtype=float)
    initial_pop = evolutionary.generate_normdist_points(starting_point, 1000, 0.05)

    plot_evolutionary_algorithm(sinxy, initial_pop, 0.25, 0.07, 500, 25)


if __name__ == "__main__":
    test_evo_algorithm()
