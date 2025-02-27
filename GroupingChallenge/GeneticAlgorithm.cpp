#include "GeneticAlgorithm.h"


GeneticAlgorithm::GeneticAlgorithm(const int& maxClusters, mt19937& randomEngine) : 
    iteration(0), maxClusters(maxClusters), randomEngine(randomEngine), bestIndividual(),
    kMeans(maxClusters, DEFAULT_KMEANS_ITERATIONS) {}

void GeneticAlgorithm::initializePointDistances(const vector<CPoint>& points) {
    if (points.empty()) {
		return;
    }
    pointDistances.resize(points.size() - 1);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i + 1 < points.size(); ++i) {
        pointDistances[i].resize(points.size() - 1 - i);
        for (size_t pointIndex = i + 1; pointIndex < points.size(); ++pointIndex) {
            pointDistances[i][pointIndex - 1 - i] = PointFunctions::calculateDistance(points[i], points[pointIndex]);
        }
    }
}

void GeneticAlgorithm::initialize(const vector<CPoint>& points) {
    initializePointDistances(points);

    const int ISLANDS_COUNT = EXPLOITATIVE_ISLANDS_COUNT + EXPLORATIVE_ISLANDS_COUNT;
    int islandPopulationSize = (ISLANDS_COUNT * ISLAND_SIZE) / ISLANDS_COUNT;

    islands.resize(ISLANDS_COUNT);
    islands[0] = new Island(IslandType::EXPLOITATIVE, islandPopulationSize, 0.02, 0.39,
        randomEngine, pointDistances, points, maxClusters);
    islands[1] = new Island(IslandType::EXPLOITATIVE, islandPopulationSize, 0.02, 0.28,
        randomEngine, pointDistances, points, maxClusters);
    islands[2] = new Island(IslandType::EXPLOITATIVE, islandPopulationSize, 0.03, 0.33,
        randomEngine, pointDistances, points, maxClusters);
    islands[3] = new Island(IslandType::EXPLORATIVE, islandPopulationSize, 0.33, 0.61,
        randomEngine, pointDistances, points, maxClusters);
    islands[4] = new Island(IslandType::EXPLORATIVE, islandPopulationSize, 0.34, 0.67,
        randomEngine, pointDistances, points, maxClusters);
    islands[5] = new Island(IslandType::EXPLORATIVE, islandPopulationSize, 0.32, 0.59,
        randomEngine, pointDistances, points, maxClusters);
    islands[6] = new Island(IslandType::EXPLORATIVE, islandPopulationSize, 0.36, 0.68,
        randomEngine, pointDistances, points, maxClusters);

	exploitativeIslands.assign(islands.begin(), islands.begin() + EXPLOITATIVE_ISLANDS_COUNT);
	explorativeIslands.assign(islands.begin() + EXPLOITATIVE_ISLANDS_COUNT, islands.end());
}


void GeneticAlgorithm::runIteration(const vector<CPoint>& points) {

    #pragma omp parallel for
    for (size_t i = 0; i < islands.size(); ++i) {
        islands[i]->runGeneration(iteration);
    }

    if (iteration % MIGRATION_FROM_EXPLOITATIVE == 0) {
        performMigrationExploitative(MIGRATION_EXPLOITATIVE_PERCENT);
    }

    if (iteration % MIGRATION_FROM_EXPLORATIVE == 0) {
        performMigrationExplorative(MIGRATION_EXPLORATIVE_PERCENT);
    }

    #pragma omp parallel for schedule(dynamic) private(randomEngine)
    for (size_t i = 0; i < islands.size(); ++i) {
        if (islands[i]->isStagnating(STAGNATION_ITERATIONS)) {
            int migrantsCount = STAGNATION_MIGRANTS_PERCENT * islands[i]->population.size();
            if (islands[i]->islandType == EXPLOITATIVE) {
                mt19937 localRandomEngine = randomEngine;
                uniform_int_distribution<int> islandIndex(0, explorativeIslands.size() - 1);
                size_t index = islandIndex(localRandomEngine);
                migrateIsland(explorativeIslands[index], islands[i], migrantsCount);
            }
            else {
                #pragma omp critical
                {
                    islands[i]->mutationProbability = max(ISLAND_MAX_MUTATION_PROBABILITY, 
                        1.027 * islands[i]->mutationProbability);
                }
            }
        }
    }

    Individual globalBest = findGlobalBest();
    applyFinalKMeans(globalBest, iteration, points);

    #pragma omp critical
    {
        if (globalBest.calculateFitness(pointDistances) < bestIndividual.calculateFitness(pointDistances)) {
            bestIndividual = globalBest;
        }
    }
    
	iteration++;
}


void GeneticAlgorithm::migrateIsland(Island* source, Island* target, const int& migrantsCount) {
    vector<Individual> migrants(migrantsCount);

    if (target->islandType == EXPLORATIVE) {
        // losowe zastepuja losowych
        #pragma omp parallel for
        for (int j = 0; j < migrantsCount; ++j) {
            int randomIndex = randomEngine() % source->population.size();
            migrants[j] = source->population[randomIndex];
        }

        #pragma omp parallel for
        for (int j = 0; j < migrantsCount; ++j) {
            int randomIndex = randomEngine() % target->population.size();
            target->population[randomIndex] = move(migrants[j]);
        }
    }
    else {
        // najlepsze zastepuja najgorsze
        partial_sort(source->population.begin(), source->population.begin() + migrantsCount, source->population.end(),
            [](const Individual& a, const Individual& b) {
                return a.fitness < b.fitness;
            });

        partial_sort(target->population.begin(), target->population.begin() + migrantsCount, target->population.end(),
            [](const Individual& a, const Individual& b) {
                return a.fitness > b.fitness;
            });

        #pragma omp parallel for
        for (size_t j = 0; j < migrantsCount; ++j) {
            target->population[j] = source->population[j];
        }

        target->kMeansForWorst(0.15);
    }

}

void GeneticAlgorithm::migrateIsland(const size_t& i, IslandType source, IslandType target,
    const int& migrantsCount) {
    vector<Individual> migrants(migrantsCount);

    Island* sourceIsland = islands[0];
    Island* targetIsland = islands[0];
    if (source == EXPLOITATIVE) {
        sourceIsland = exploitativeIslands[i];
        if (target == EXPLORATIVE) {
            targetIsland = explorativeIslands[randIslandIndex(IslandType::EXPLORATIVE)];
        }
        else {
            targetIsland = exploitativeIslands[randIslandIndex(IslandType::EXPLOITATIVE, i)];
        }
    }
    else {
        sourceIsland = explorativeIslands[i];
        if (target == EXPLORATIVE) {
            targetIsland = explorativeIslands[randIslandIndex(IslandType::EXPLORATIVE, i)];
        }
        else {
            targetIsland = exploitativeIslands[randIslandIndex(IslandType::EXPLOITATIVE)];
        }
    }
    migrateIsland(sourceIsland, targetIsland, migrantsCount);
}

size_t GeneticAlgorithm::randIslandIndex(IslandType islandType, const int excludedIndex) const {
    int maxIndex = islandType == EXPLORATIVE ? explorativeIslands.size() - 1 : 
        exploitativeIslands.size() - 1;

    uniform_int_distribution<int> distrIslandIndex(0, maxIndex);

    size_t nextIslandIndex;
    do {
        nextIslandIndex = distrIslandIndex(randomEngine);
    } while (nextIslandIndex == static_cast<size_t>(excludedIndex));
    return nextIslandIndex;
}

void GeneticAlgorithm::performMigrationExploitative(const double& migratedPopulationPercent) {
    int migrantsCount = migratedPopulationPercent * exploitativeIslands[0]->population.size();
    for (size_t i = 0; i < exploitativeIslands.size(); ++i) {
        IslandType target;
        uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(randomEngine) <= 0.965) {
            target = EXPLORATIVE;
        }
        else {
            target = EXPLOITATIVE;
        }

        migrateIsland(i, EXPLOITATIVE, target, migrantsCount);
    }
}

void GeneticAlgorithm::performMigrationExplorative(const double& migratedPopulationPercent) {
    int migrantsCount = migratedPopulationPercent * explorativeIslands[0]->population.size();
    for (size_t i = 0; i < explorativeIslands.size(); ++i) {
        IslandType target;
        uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(randomEngine) <= 0.11) {
            target = EXPLORATIVE;
        }
        else {
            target = EXPLOITATIVE;
        }

        migrateIsland(i, EXPLORATIVE, target, migrantsCount);
    }
}

Individual GeneticAlgorithm::findGlobalBest() {
    Individual globalBest = islands[0]->bestIndividual;
    for (auto& island : islands) {
        if (island->bestIndividual.calculateFitness(pointDistances) < globalBest.calculateFitness(pointDistances)) {
            globalBest = island->bestIndividual;
        }
    }
    return globalBest;
}

void GeneticAlgorithm::applyFinalKMeans(Individual& bestIndividual, const size_t& iteration, 
    const vector<CPoint>& points) {
    if (iteration > 500) {
        kMeans.setMaxIterations(30);
    }
    Individual improved = kMeans.getIndividual(*bestIndividual.getGenotype(), points);

    if (improved.calculateFitness(pointDistances) < bestIndividual.calculateFitness(pointDistances)) {
        bestIndividual = improved;
    }
}