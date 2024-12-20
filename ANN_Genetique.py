import numpy as np
import tensorflow as tf
from tqdm import tqdm

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
X_train = X_train[:500,:]
y_train = y_train[:500,:]

class NeuralNetwork:
    def __init__(self, architecture):
        self.architecture = architecture
        self.weights = []
        self.bias = []
        self.layers = len(architecture)
        self.fitness = 0
        for i in range(1, self.layers):
            self.weights.append(np.random.randn(architecture[i - 1], architecture[i]))
            self.bias.append(np.random.randn(1, architecture[i]))

    def feedforward(self, X):
        a = X
        for i in range(self.layers - 2):
            z = np.dot(a, self.weights[i]) + self.bias[i]
            a = 1 / (1 + np.exp(-z))
        z = np.dot(a, self.weights[-1]) + self.bias[-1]
        a = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        return a

    def mutate(self, rate):
        for i in range(len(self.weights)):
            mask = np.random.rand(*self.weights[i].shape) < rate
            self.weights[i][mask] += np.random.normal(0, 0.1, np.count_nonzero(mask))
            mask = np.random.rand(*self.bias[i].shape) < rate
            self.bias[i][mask] += np.random.normal(0, 0.1, np.count_nonzero(mask))

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, elite_size, generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.generations = generations

    def initialize_population(self, architecture):
        population = []
        for i in range(self.population_size):
            population.append(NeuralNetwork(architecture))
        return population

    def evaluate_fitness(self, population, X, y):
        for network in population:
            predictions = network.feedforward(X)
            network.fitness = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))

    def select_parents(self, population):
        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        fitness_sum = sum([network.fitness for network in population[self.elite_size:]])
        parents = population[:self.elite_size]
        for i in range(self.population_size - self.elite_size):
            prob = [network.fitness / fitness_sum for network in population[self.elite_size:]]
            parent = np.random.choice(population[self.elite_size:], size=1, p=prob)[0]
            parents.append(parent)
        return parents

    def crossover(self, parent1, parent2):
        child = NeuralNetwork(parent1.architecture)
        for i in range(len(parent1.weights)):
            if i >= len(child.weights):
                child.weights.append(np.zeros_like(parent1.weights[i]))
                child.bias.append(np.zeros_like(parent1.bias[i]))
            mask = np.random.rand(*parent1.weights[i].shape) < 0.5
            child.weights[i][mask] = parent1.weights[i][mask]
            child.weights[i][~mask] = parent2.weights[i][~mask]
            mask = np.random.rand(*parent1.bias[i].shape) < 0.5
            child.bias[i][mask] = parent1.bias[i][mask]
            child.bias[i][~mask] = parent2.bias[i][~mask]
        return child

    def create_child(self, parents):
        child = []
        for i in range(self.population_size - self.elite_size):
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)
            child.append(self.crossover(parent1, parent2))
        return child

    def mutate_population(self, population):
        for network in population:
            network.mutate(self.mutation_rate)

    def run(self, X, y, architecture):
        population = self.initialize_population(architecture)
        for i in tqdm(range(self.generations)):
            self.evaluate_fitness(population, X, y)
            parents = self.select_parents(population)
            child = self.create_child(parents)
            population = parents + child
            self.mutate_population(population)
            #print("Generation: {}/{}".format(i + 1, self.generations))
        self.evaluate_fitness(population, X, y)
        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        return population[0]

def affiche():
    genetic_algorithm = GeneticAlgorithm(population_size=50, mutation_rate=0.1, elite_size=25, generations=800)
    best_network = genetic_algorithm.run(X_train, y_train, [784,32, 10])
    predictions = best_network.feedforward(X_test)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
    print("Accuracy: {:.2f}%".format(accuracy * 100))

affiche()