#include "Individual.h"
#include "GroupingEvaluator.h"

#include <iostream>

Individual::Individual(const int& numberOfPoints, const int& maxClusters, mt19937& randomEngine)
	: numberOfPoints(numberOfPoints), maxClusters(maxClusters), fitness(0.0), isFitnessCurrent(false) {
	uniform_int_distribution<int> distribution(1, maxClusters);
	genotype.resize(numberOfPoints);
	for (int& gene : genotype) {
		gene = distribution(randomEngine);
	}
	groups.resize(maxClusters + 1);
}

Individual::Individual(const vector<int>& genotype, const int& maxClusters)
	: maxClusters(maxClusters), fitness(0.0), isFitnessCurrent(false) {
	numberOfPoints = genotype.size();
	this->genotype.resize(numberOfPoints);
	this->genotype = genotype;
	groups.resize(maxClusters + 1);

}

void Individual::rebuildGroups() {
	for (auto& group : groups) {
		group.clear(); 
	}
	for (size_t i = 0; i < numberOfPoints; ++i) {
		groups[genotype[i]].push_back(i);
	}
}

double Individual::getPointDistance(const size_t& sourcePointIndex, const size_t& targetPointIndex,
	const vector<vector<double>>& pointDistances) const {
	if (sourcePointIndex < targetPointIndex) {
		return pointDistances[sourcePointIndex][targetPointIndex - 1 - sourcePointIndex];
	}
	else if (targetPointIndex < sourcePointIndex) {
		return pointDistances[targetPointIndex][sourcePointIndex - 1 - targetPointIndex];
	}
	return 0.0;
}


double Individual::calculateFitness(const vector<vector<double>>& pointDistances) {
	if (!isFitnessCurrent) {
		rebuildGroups();

		double totalFitness = 0.0;

		#pragma omp parallel for reduction(+:totalFitness) schedule(dynamic)
		for (size_t clusterIndex = 1; clusterIndex <= maxClusters; ++clusterIndex) {
			const vector<size_t>& indices = groups[clusterIndex];
			for (size_t i = 0; i + 1 < indices.size(); ++i) {
				for (size_t j = i + 1; j < indices.size(); ++j) {
					totalFitness += getPointDistance(indices[i], indices[j], pointDistances);
				}
			}
		}

		fitness = totalFitness;
		isFitnessCurrent = true;
	}
	return fitness;
}

double Individual::updateFitnessForGeneChange(const int& geneIndex, const int& oldCluster, const int& newCluster,
	const vector<vector<double>>& pointDistances) {
	if (!isFitnessCurrent) {
		return calculateFitness(pointDistances);
	}

	if (oldCluster == newCluster) {
		return fitness;
	}

	const auto& oldGroup = groups[oldCluster];
	const auto& newGroup = groups[newCluster];

	#pragma omp parallel for reduction(-:fitness) schedule(dynamic)
	for (size_t i = 0; i < oldGroup.size(); ++i) {
		if (oldGroup[i] != geneIndex) {
			fitness -= getPointDistance(geneIndex, oldGroup[i], pointDistances);
		}
	}

	#pragma omp parallel for reduction(+:fitness) schedule(dynamic)
	for (size_t i = 0; i < newGroup.size(); ++i) {
		if (newGroup[i] != geneIndex) {
			fitness += getPointDistance(geneIndex, newGroup[i], pointDistances);
		}
	}

	groups[oldCluster].erase(remove(groups[oldCluster].begin(), groups[oldCluster].end(), geneIndex),
		groups[oldCluster].end());
	groups[newCluster].push_back(geneIndex);

	return fitness;
}

pair<Individual, Individual> Individual::onePointCrossover(const Individual& other, mt19937& randomEngine) const {
	int pointIndex = randomEngine() % numberOfPoints;
	vector<int> genotype1(this->genotype);
	vector<int> genotype2(this->genotype);

	for (size_t i = 0; i < pointIndex; ++i) {
		genotype2[i] = other.genotype[i];
	}
	for (size_t i = pointIndex; i < numberOfPoints; ++i) {
		genotype1[i] = other.genotype[i];
	}
	return move(make_pair(Individual(move(genotype1), this->maxClusters),
		Individual(move(genotype2), this->maxClusters)));
}

pair<Individual, Individual> Individual::multiPointCrossover(const Individual& other, mt19937& randomEngine) const {
	int pointIndex1 = randomEngine() % numberOfPoints;
	int pointIndex2 = randomEngine() % numberOfPoints;
	if (pointIndex1 > pointIndex2) {
		swap(pointIndex1, pointIndex2);
	}

	vector<int> genotype1(this->genotype);
	vector<int> genotype2(other.genotype);
	for (int i = pointIndex1; i <= pointIndex2; ++i) {
		swap(genotype1[i], genotype2[i]);
	}

	return move(make_pair(Individual(move(genotype1), this->maxClusters),
		Individual(move(genotype2), this->maxClusters)));
}

pair<Individual, Individual> Individual::uniformCrossover(const Individual& other, mt19937& randomEngine) const {
	vector<int> genotype1(this->genotype);
	vector<int> genotype2(other.genotype);
	
	#pragma omp parallel for
	for (size_t i = 0; i < numberOfPoints; ++i) {
		if (randomEngine() % 2) {
			swap(genotype1[i], genotype2[i]);
		}
	}
	return move(make_pair(Individual(move(genotype1), this->maxClusters),
		Individual(move(genotype2), this->maxClusters)));
}

void Individual::mutate(double mutationProbability, mt19937& randomEngine, const vector<vector<double>>& pointDistances) {
	uniform_real_distribution<double> probability(0.0, 1.0);
	uniform_int_distribution<int> clusterDist(1, maxClusters);

	#pragma omp parallel for
	for (size_t i = 0; i < numberOfPoints; ++i) {
		if (probability(randomEngine) < mutationProbability) {
			int oldCluster = genotype[i];
			int newCluster = clusterDist(randomEngine);

			if (oldCluster != newCluster) {
				#pragma omp critical
				{
					genotype[i] = newCluster;
					updateFitnessForGeneChange(i, oldCluster, newCluster, pointDistances);
				}
			}
		}
	}
}

void Individual::adaptiveMutate(mt19937& randomEngine, const vector<vector<double>>& pointDistances) {
	uniform_int_distribution<int> clusterDist(1, maxClusters);

	calculateFitness(pointDistances);

	#pragma omp parallel for
	for (size_t i = 0; i < numberOfPoints; ++i) {
		int oldCluster = genotype[i];
		int newCluster = clusterDist(randomEngine);

		if (oldCluster != newCluster) {
			double oldFitness = fitness;

			#pragma omp critical
			{
				genotype[i] = newCluster;
				updateFitnessForGeneChange(i, oldCluster, newCluster, pointDistances);
			}
			if (oldFitness < fitness) {
				#pragma omp critical
				{
					genotype[i] = oldCluster;
					updateFitnessForGeneChange(i, newCluster, oldCluster, pointDistances);
				}
			}
		}
	}
}