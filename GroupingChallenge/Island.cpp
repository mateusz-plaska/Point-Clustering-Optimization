#include "Island.h"
#include "KMeans.h"


Island::Island(IslandType islandType, const int& populationSize, const double& mutationProb, 
    const double& crossoverProb, mt19937& randomEngine, const vector<vector<double>>& pointDistances,
    const vector<CPoint>& points, const int& maxClusters) : mutationProbability(mutationProb), 
    crossoverProbability(crossoverProb), randomEngine(randomEngine), maxClusters(maxClusters), 
    islandType(islandType), noImprovementCounter(0), points(points), pointDistances(pointDistances),
    kMeans(maxClusters, 2) {

    population.reserve(populationSize);
    for (int i = 0; i < populationSize; ++i) {
        population.emplace_back(Individual(points.size(), maxClusters, randomEngine));
    }
    fitness.resize(populationSize);
    setInitialRandomIndividualsPercent();
    evaluatePopulation();
}

void Island::runGeneration(const size_t& iteration) {
    if (islandType == EXPLOITATIVE) {
        runExploitative(iteration);
    }
    else {
        runExplorative(iteration);
    }
}

void Island::runExploitative(const size_t& iteration) {
    vector<Individual> newPopulation;
    newPopulation.reserve(population.size());

    vector<pair<double, size_t>> fitnessIndices;
	fitnessIndices.reserve(fitness.size());
    #pragma omp parallel for
    for (size_t i = 0; i < fitness.size(); ++i) {
        fitnessIndices.push_back({ fitness[i], i });
    }
    int eliteCount = static_cast<int>(ELITE_RATE_EXPLOITATIVE * population.size());
    if (eliteCount % 2) {
        eliteCount--;
    }

    partial_sort(fitnessIndices.begin(), fitnessIndices.begin() + eliteCount, fitnessIndices.end(),
        [](const pair<double, size_t>& p1, const pair<double, size_t>& p2) {
            return p1.first < p2.first;
        });

    #pragma omp parallel for
    for (size_t i = 0; i < population.size(); ++i) {
        size_t index = fitnessIndices[i].second;
        double mutationChance = (i < eliteCount) ? BETTER_INDIVIDUALS_CHANCE : REST_INDIVIDUALS_CHANCE;
        uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(randomEngine) < mutationChance) {
            population[index].adaptiveMutate(randomEngine, pointDistances);
        }
    }

    #pragma omp paraller for
    for (size_t i = 0; i < eliteCount; ++i) {
        #pragma omp critical
        newPopulation.push_back(population[fitnessIndices[i].second]);
    }


    int randomIndividualsCount = randomIndividualsPercent * population.size();
    if (randomIndividualsCount % 2) {
        randomIndividualsCount--;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < randomIndividualsCount; ++i) {
        Individual randomIndividual = Individual(points.size(), maxClusters, randomEngine);
        randomIndividual.calculateFitness(pointDistances);
        #pragma omp critical
        newPopulation.push_back(randomIndividual);
    }

    int endIndex = population.size() - randomIndividualsCount;

    #pragma omp parallel for
    for (size_t i = eliteCount; i < endIndex; i += 2) {
        Individual& parent1 = population[i];
        Individual& parent2 = population[i];

        uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(randomEngine) <= ROULETTE_SELECTION_EXPLOITATIVE) {
            parent1 = this->rouletteSelection();
            parent2 = this->rouletteSelection();
        }
        else {
            parent1 = this->tournamentSelection(3);
            parent2 = this->tournamentSelection(3);
        }


        if (dist(randomEngine) < crossoverProbability) {
            auto offspring = parent1.multiPointCrossover(parent2, randomEngine);
            parent1 = move(offspring.first);
            parent2 = move(offspring.second);
        }

        parent1.mutate(mutationProbability, randomEngine, pointDistances);
        parent2.mutate(mutationProbability, randomEngine, pointDistances);


        double adaptiveProbability = min(MAX_ADAPTIVE_PROB_EXPLOITATIVE, mutationProbability +
            (parent1.calculateFitness(pointDistances) / fitness[maxFitnessIndex]));
        if (dist(randomEngine) < adaptiveProbability) {
            parent1.adaptiveMutate(randomEngine, pointDistances);
        }
        adaptiveProbability = min(MAX_ADAPTIVE_PROB_EXPLOITATIVE, mutationProbability +
            (parent2.calculateFitness(pointDistances) / fitness[maxFitnessIndex]));
        if (dist(randomEngine) < adaptiveProbability) {
            parent2.adaptiveMutate(randomEngine, pointDistances);
        }

        #pragma omp critical
        {
            newPopulation.push_back(parent1);
            newPopulation.push_back(parent2);
        }
    }

    population = move(newPopulation);

    if ((iteration + 1) % KMEANS_ITERATION_EXPLOITATIVE == 0) {
        kMeansExploitative(KMEANS_PERCENT_EXPLOITATIVE);
    }

    evaluatePopulation();
}

void Island::kMeansExploitative(const double& populationPercent) {
    #pragma omp parallel for
    for (size_t i = 0; i < populationPercent * population.size(); ++i) {
        uniform_int_distribution<int> dist(0, population.size() - 1);
        size_t index = dist(randomEngine);
        Individual improved = kMeans.getIndividual(*population[index].getGenotype(), points);
        if (improved.calculateFitness(pointDistances) < population[index].calculateFitness(pointDistances)) {
            #pragma omp critical
            population[index] = move(improved);
        }
    }
}

void Island::kMeansForWorst(const double& populationPercent) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = population.size() - populationPercent * population.size(); i < population.size(); ++i) {
        Individual improved = kMeans.getIndividual(*population[i].getGenotype(), points);       
        if (improved.calculateFitness(pointDistances) < population[i].calculateFitness(pointDistances)) {
            #pragma omp critical
            population[i] = move(improved);
        }
    }
}


void Island::runExplorative(const size_t& iteration) {
    vector<Individual> newPopulation;
    newPopulation.reserve(population.size());

    int randomIndividualsCount = randomIndividualsPercent * population.size();
    if (randomIndividualsCount % 2) {
        randomIndividualsCount++;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < randomIndividualsCount; ++i) {
        Individual randomIndividual = Individual(points.size(), maxClusters, randomEngine);
        randomIndividual.calculateFitness(pointDistances);
        #pragma omp critical
        newPopulation.push_back(randomIndividual);
    }

    int endIndex = population.size() - randomIndividualsCount;

    #pragma omp parallel for
    for (size_t i = 0; i < endIndex; i += 2) {
        Individual& parent1 = population[i];
        Individual& parent2 = population[i];

        uniform_real_distribution<double> distr(0.0, 1.0);

        if (distr(randomEngine) < RANDOM_SELECTION_EXPLORATIVE) {
            uniform_int_distribution<int> intDist(0, population.size() - 1);
            parent1 = population[intDist(randomEngine)];
            parent2 = population[intDist(randomEngine)];
        }
        else {
            parent1 = this->rouletteSelectionWorst();
            parent2 = this->rouletteSelectionWorst();
        }

        if (distr(randomEngine) < crossoverProbability) {
            auto offspring = parent1.uniformCrossover(parent2, randomEngine);
            parent1 = move(offspring.first);
            parent2 = move(offspring.second);
        }

        parent1.mutate(mutationProbability, randomEngine, pointDistances);
        parent2.mutate(mutationProbability, randomEngine, pointDistances);

        if (distr(randomEngine) < ADAPTIVE_PROB_EXPLORATIVE) {
            parent1.adaptiveMutate(randomEngine, pointDistances);
        }
        if (distr(randomEngine) < ADAPTIVE_PROB_EXPLORATIVE) {
            parent2.adaptiveMutate(randomEngine, pointDistances);
        }

        #pragma omp critical
        {
            newPopulation.push_back(parent1);
            newPopulation.push_back(parent2);
        }
    }

    population = move(newPopulation);

    evaluatePopulation();
}

bool Island::isStagnating(const size_t& noImprovementThreshold) {
    if (noImprovementCounter > noImprovementThreshold) {
        noImprovementCounter = 0;
        return true;
    }
    return false;
}

void Island::evaluatePopulation() {
    #pragma omp parallel for
    for (int i = 0; i < population.size(); ++i) {
        fitness[i] = population[i].calculateFitness(pointDistances);
    }
    int bestIndex = min_element(fitness.begin(), fitness.end()) - fitness.begin();
    maxFitnessIndex = max_element(fitness.begin(), fitness.end()) - fitness.begin();

    #pragma omp critical
    {
        if (fitness[bestIndex] < bestIndividual.calculateFitness(pointDistances)) {
            bestIndividual = population[bestIndex];
            noImprovementCounter = 0;
        }
        else {
            noImprovementCounter++;
        }
    }
}

Individual& Island::tournamentSelection(int amountOfDraw) {
    int bestIndex = randomEngine() % population.size();
    for (int i = 1; i < amountOfDraw; ++i) {
        int candidate = randomEngine() % population.size();
        if (fitness[candidate] < fitness[bestIndex]) {
            bestIndex = candidate;
        }
    }
    return population[bestIndex];
}

Individual& Island::rouletteSelection() {
    double totalFitness = 0.0;
    for (double f : fitness) {
        totalFitness += f;
    }

    uniform_real_distribution<> dist(0, 1);
    double target = dist(randomEngine);
    double cumulative = 0.0;
    for (size_t i = 0; i < population.size(); ++i) {
        cumulative += fitness[i] / totalFitness;
        if (cumulative >= target) {
            return population[i];
        }
    }
    return population.back();
}

Individual& Island::rouletteSelectionWorst() {
    double totalInvertedFitness = 0.0;
    for (double f : fitness) {
        totalInvertedFitness += 1.0 / (f + 1e-10);
    }

    uniform_real_distribution<> dist(0, 1);
    double target = dist(randomEngine);
    double cumulative = 0.0;
    for (size_t i = 0; i < population.size(); ++i) {
        double invertedFitness = 1.0 / (fitness[i] + 1e-10);
        cumulative += (invertedFitness / totalInvertedFitness);
        if (cumulative >= target) {
            return population[i];
        }
    }
    return population.back();
}