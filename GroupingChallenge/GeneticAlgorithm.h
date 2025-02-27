#pragma once

#include "Island.h"
#include "KMeans.h"

constexpr int DEFAULT_KMEANS_ITERATIONS = 6;
constexpr int EXPLOITATIVE_ISLANDS_COUNT = 3;
constexpr int EXPLORATIVE_ISLANDS_COUNT = 4;
constexpr int ISLAND_SIZE = 100;

constexpr int MIGRATION_FROM_EXPLOITATIVE = 18;         
constexpr int MIGRATION_FROM_EXPLORATIVE = 5;
constexpr int STAGNATION_ITERATIONS = 9;    

constexpr double MIGRATION_EXPLOITATIVE_PERCENT = 0.12;
constexpr double MIGRATION_EXPLORATIVE_PERCENT = 0.17;

constexpr double ISLAND_MAX_MUTATION_PROBABILITY = 0.77;
constexpr double STAGNATION_MIGRANTS_PERCENT = 0.15;


class GeneticAlgorithm
{
public:
    GeneticAlgorithm(const int& maxClusters, mt19937& randomEngine);

    ~GeneticAlgorithm() {
        for (Island* island : islands) {
            delete island;
        }
    }

    void initialize(const vector<CPoint>& points);
    void runIteration(const vector<CPoint>& points);
    
    const Individual& getBestIndividual() const {
        return bestIndividual;
    }

private:
	int iteration;
	int maxClusters;
    Individual bestIndividual;
    mt19937& randomEngine;

    KMeans kMeans;

    vector<vector<double>> pointDistances;

    vector<Island*> islands;
    vector<Island*> explorativeIslands;
    vector<Island*> exploitativeIslands;


    void initializePointDistances(const vector<CPoint>& points);

    void migrateIsland(Island* source, Island* target, const int& migrantsCount);
    void migrateIsland(const size_t& islandIndex, IslandType source, IslandType target, const int& migrantsCount);
    void performMigrationExploitative(const double& migratedPopulationPercent);
    void performMigrationExplorative(const double& migratedPopulationPercent);
  
    size_t randIslandIndex(IslandType islandType, const int excludedIndex = -1) const;

    Individual findGlobalBest();
    void applyFinalKMeans(Individual& bestIndividual, const size_t& iteration, const vector<CPoint>& points);
};

