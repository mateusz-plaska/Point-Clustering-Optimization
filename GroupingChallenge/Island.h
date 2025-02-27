#pragma once

#include "Individual.h"
#include "KMeans.h"

const enum IslandType { EXPLORATIVE, EXPLOITATIVE };

constexpr double ELITE_RATE_EXPLOITATIVE = 0.04;
constexpr double BETTER_INDIVIDUALS_CHANCE = 0.29;
constexpr double REST_INDIVIDUALS_CHANCE = 0.205;        
constexpr double ROULETTE_SELECTION_EXPLOITATIVE = 0.325; 
constexpr double MAX_ADAPTIVE_PROB_EXPLOITATIVE = 0.4;
constexpr int KMEANS_ITERATION_EXPLOITATIVE = 18;
constexpr double KMEANS_PERCENT_EXPLOITATIVE = 0.2;

constexpr double RANDOM_SELECTION_EXPLORATIVE = 0.4;
constexpr double ADAPTIVE_PROB_EXPLORATIVE = 0.17;


class Island {
public:
    Island(IslandType islandType, const int& populationSize, const double& mutationProb, 
        const double& crossoverProb, mt19937& randomEngine, const vector<vector<double>>& pointDistances,
        const vector<CPoint>& points, const int& maxClusters);

    void runGeneration(const size_t& iteration);

    void kMeansForWorst(const double& populationPercent);
   
    bool isStagnating(const size_t& noImprovementThreshold);

    vector<Individual> population;        
    Individual bestIndividual;            
    IslandType islandType;

    double mutationProbability;           

private:
    int noImprovementCounter;         
    double crossoverProbability;          

    double randomIndividualsPercent;
    int maxFitnessIndex;

    vector<double> fitness;
    vector<vector<double>> pointDistances;

    mt19937 randomEngine;

    int maxClusters;
    vector<CPoint> points;
	KMeans kMeans;


    void runExploitative(const size_t& iteration);
    void kMeansExploitative(const double& populationPercent);
    void runExplorative(const size_t& iteration);

    void evaluatePopulation();

    Individual& tournamentSelection(int amountOfDraw);
    Individual& rouletteSelection();
    Individual& rouletteSelectionWorst();

    void setInitialRandomIndividualsPercent() {
        islandType == EXPLORATIVE ? randomIndividualsPercent = 0.15 : randomIndividualsPercent = 0.06;
    }
};