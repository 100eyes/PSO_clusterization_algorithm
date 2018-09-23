from tqdm import tqdm
from math import sqrt
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

iterations = 200
swarm_size = 10
num_clusters = 4

w = 0.729844  # Inertia weight to prevent velocities becoming too large
c1 = 1.496180  # Scaling co-efficient on the social component
c2 = 1.496180  # Scaling co-efficient on the cognitive component
w_damp = 0.99   # Damping factor for velocity of particle

global_best_position = None
global_best_fitness = None

w1 = 0.5    # Fitness coefficient
w2 = 0.5    # Fitness coefficient
z_max = 255*sqrt(3)


class Particle(object):
    def __init__(self, number_clusters, im_shape):
        self.fitness = float
        self.position = np.random.randint(0, 256, (number_clusters, im_shape[2]))
        self.best_fitness = None
        self.best_position = self.position
        self.velocity = 0.0
        self.clusters = []
        self.num_clusters = number_clusters
        self.im_shape = im_shape

    def update_clusters(self, image):
        self.clusters = [np.zeros((self.im_shape[0], self.im_shape[1]), dtype=np.bool) for _ in range(self.num_clusters)]
        dim_x = image.shape[0]
        dim_y = image.shape[1]
        for i in range(dim_x):
            for j in range(dim_y):
                minimum = sqrt(((image[i, j, :] - self.position[0, :])**2).sum())
                num_of_cluster = 0
                for k in range(1, self.num_clusters):
                    distance = sqrt(((image[i, j, :] - self.position[k, :])**2).sum())
                    if distance < minimum:
                        minimum = distance
                        num_of_cluster = k
                self.clusters[num_of_cluster][i, j] = True

    def d_max(self, image):
        maximum = 0
        for i in range(self.num_clusters):
            if self.clusters[i].sum() == 0:
                continue
            pixels_in_cluster = image[np.where(self.clusters[i])]
            d_prvo = (pixels_in_cluster-self.position[i, :])**2
            d_prvo = d_prvo.reshape((d_prvo.shape[0], d_prvo.shape[1], 1))
            d_prvo = np.sqrt(d_prvo.sum(axis=2)).sum().sum()
            d = d_prvo / self.clusters[i].sum()
            if d > maximum:
                maximum = d
        return maximum

    def d_min(self):
        minimum = float
        for i in range(self.num_clusters):
            for j in range(i + 1, self.num_clusters):
                p1 = self.position[i, :]
                p2 = self.position[j, :]
                d = np.sqrt(((p1 - p2) ** 2).sum())
                if i == 0 and j == 1:
                    minimum = d
                else:
                    if d < minimum:
                        minimum = d
        return minimum

    def calculate_fitness(self, image):
        global w1, w2, z_max
        self.fitness = w1*self.d_max(image) + w2*(z_max - self.d_min())
        if self.best_fitness is None:
            self.best_fitness = self.fitness
        elif self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position

    def delete_clusters(self):
        self.clusters = []

    def update_position(self):
        global w, c1, c2, global_best_position, w_damp
        r1 = c1*np.random.random((self.num_clusters, 1))
        r2 = c2*np.random.random((self.num_clusters, 1))
        self.velocity = w*self.velocity + \
                        r1*(self.best_position - self.position) + \
                        r2*(global_best_position - self.position)
        self.position = self.position + self.velocity
        w *= w_damp


def main():
    slika = mpimg.imread("/home/besim/Documents/Fax/OR/projekat/Lenna.jpg")
    if slika.ndim < 3:
        slika = slika.reshape((slika.shape[0], slika.shape[1], 1))
    print(slika.shape)
    global swarm_size, global_best_position, global_best_fitness, iterations, num_clusters
    particles = [Particle(num_clusters, im_shape=slika.shape) for _ in range(swarm_size)]
    particles[0].update_clusters(slika)
    particles[0].calculate_fitness(slika)
    global_best_fitness = particles[0].fitness
    global_best_position = particles[0].position

    for i in tqdm(range(iterations)):
        for particle in particles:
            particle.update_clusters(slika)
            particle.calculate_fitness(slika)
            particle.delete_clusters()
            if particle.fitness < global_best_fitness:
                global_best_fitness = particle.fitness
                global_best_position = particle.position

        for particle in particles:
            particle.update_position()
    particles[0].position=global_best_position
    particles[0].update_clusters(slika)
    for i in range(num_clusters):
        cluster = particles[0].clusters[i]
        np.save("/home/besim/Documents/Fax/OR/projekat/cluster"+str(i), cluster)
        plt.figure()
        # cluster = cluster.reshape((cluster.shape[0], cluster.shape[1], 1))
        # cluster = cluster*slika
        plt.imshow(cluster)

    plt.show()

main()



