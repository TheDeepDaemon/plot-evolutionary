import numpy as np


def calc_expected_population_delta(population, population_average, fitness_function):
    if len(population) == 0:
        return None
    total = np.zeros_like(population[0])
    for member in population:
        total += (member - population_average) * fitness_function(member)
    return total / np.linalg.norm(total)


def get_dtype_of_array(arr):
    dtype = arr.dtype
    if (
        (dtype == np.float64) or
        (dtype == np.float32) or
        (dtype == np.float16)):
        return 'float'
    elif (
        (dtype == np.int64) or
        (dtype == np.int32) or
        (dtype == np.int16) or
        (dtype == np.int8) or
        (dtype == np.uint64) or
        (dtype == np.uint32) or
        (dtype == np.uint16) or
        (dtype == np.uint8)):
        return 'int'
    elif (dtype == np.bool8):
        return 'bool'
    else:
        return None


def mutate_phenotype(population, index, mutation_point, mutation_stddev):
    # Phenotypes are usually most accurately described by
    # normally distributed real numbers,
    # any changes will probably have an additive effect.
    # A mutation might affect some traits value
    # so it is higher or lower than it would have otherwise been,
    # but the mutation will not dictate its value in an absolute sense.
    new_value = np.random.normal(scale=mutation_stddev)
    population[index][mutation_point] += new_value


def mutate_genotype(population, index, mutation_point, mutation_stddev, mutation_range, gene_datatype):
    new_value = None
    if gene_datatype == 'int':
        new_value = np.random.randint(0, mutation_range)
    elif gene_datatype == 'bool':
        new_value = bool(np.random.randint(0, 2))
    else:
        # else, assume it is a float
        new_value = np.random.normal(scale=mutation_stddev)
    
    # Genotypes are most accurately described by
    # discrete values that can be updated to new values.
    population[index][mutation_point] = new_value


def create_next_generation(population_shape, parents):
    # Create new population by recombining parents
    new_population = []
    for _ in range(population_shape[0]):
        parent1_index = np.random.randint(len(parents))
        parent2_index = np.random.randint(len(parents))
        crossover_point = np.random.randint(population_shape[1])
        child = np.concatenate((parents[parent1_index][:crossover_point], parents[parent2_index][crossover_point:]))
        new_population.append(child)
    return new_population


def apply_mutations(new_population, mutation_rate, is_phenotype, mutation_stddev, mutation_range, gene_datatype):
    # Add mutations to the new population
    for i in range(len(new_population)):
        if np.random.random() < mutation_rate:
            mutation_point = np.random.randint(len(new_population[i]))
            
            if is_phenotype:
                mutate_phenotype(new_population, i, mutation_point, mutation_stddev)
            else:
                mutate_genotype(new_population, i, mutation_point, mutation_stddev, mutation_range, gene_datatype)


# if is_phenotype, it is a vector of phenotypes, and changes will have an additive effect
# if not, then it is a vector of genes, and mutations occurring mean that specific genes have changed
def evolutionary_step(population, fitness_function, mutation_rate, mutation_stddev, mutation_range=None, is_phenotype=True):
    population_shape = population.shape
    
    # Evaluate fitness of each population member
    reproduction_probabilities = np.array([fitness_function(member) for member in population])

    # Select parents for the next generation by eliminating based on survival probability
    parents = []
    for i in range(len(population)):
        if np.random.rand() < reproduction_probabilities[i]:
            parents.append(population[i])

    # If none survived this round of selection, just return the original population
    if len(parents) == 0:
        print("The entire population was wiped out.")
        return population

    new_population = create_next_generation(population_shape, parents)
    new_population = np.array(new_population, dtype=population.dtype)

    apply_mutations(new_population, mutation_rate, is_phenotype, mutation_stddev, mutation_range, get_dtype_of_array(population))
    
    return new_population


def evolve(population, fitness_function, mutation_rate, mutation_stddev, n_steps):
    for _ in range(n_steps):
        population = evolutionary_step(population, fitness_function, mutation_rate, mutation_stddev)
    return population


def generate_normdist_points(base, num_points, std_dev):
    # Generate normally distributed random numbers
    x = np.random.normal(0, std_dev, num_points) + base[0]
    y = np.random.normal(0, std_dev, num_points) + base[1]

    points = [(x[i], y[i]) for i in range(num_points)]
    
    return np.array(points, dtype=float)


def test_evo_step():
    initial_pop_ave = (1, 1)
    initial_pop = generate_normdist_points(initial_pop_ave, 10, 1)
    
    print(initial_pop)
    
    def example_fitfunc(genes):
        return np.linalg.norm(genes)

    next_pop = evolutionary_step(
        initial_pop,
        fitness_function=example_fitfunc,
        mutation_rate=0.2,
        mutation_stddev=0.1,
        mutation_range=10,
        is_phenotype=True
    )
    
    print(next_pop)


if __name__ == "__main__":
    test_evo_step()
