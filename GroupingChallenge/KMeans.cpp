#include "KMeans.h"

#include <limits>

Individual KMeans::getIndividual(const vector<CPoint>& points, mt19937& randomEngine) {
	uniform_int_distribution<int> distribution(1, maxClusters);
	vector<int> genotype(points.size());
	for (int& gene : genotype) {
		gene = distribution(randomEngine);
	}
	return this->getIndividual(genotype, points);
}

Individual KMeans::getIndividual(const vector<int>& genotypeOriginal, const vector<CPoint>& points) {
	vector<int> genotype = genotypeOriginal;
	int numberOfPoints = points.size();

	vector<CPoint> centroids(maxClusters);
	for (size_t i = 0; i < maxClusters; ++i) {
		PointFunctions::assignSourcePointToEmpty(centroids[i], points[i % numberOfPoints]);
	}

	for (size_t iteration = 0; iteration < maxIterations; ++iteration) {
		updateCentroids(centroids, genotype, points);

		bool changed = false;

		#pragma omp parallel for reduction(|:changed)
		for (size_t i = 0; i < numberOfPoints; ++i) {
			double minDistance = numeric_limits<double>::max();
			int bestCluster = 0;

			for (int j = 0; j < maxClusters; ++j) {
				double distance = PointFunctions::calculateDistance(points[i], centroids[j]);
				if (distance < minDistance) {
					minDistance = distance;
					bestCluster = j + 1;
				}
			}

			if (genotype[i] != bestCluster) {
				genotype[i] = bestCluster;
				changed = true;
			}
		}

		if (!changed) break;
	}

	return Individual(genotype, maxClusters);
}

void KMeans::updateCentroids(vector<CPoint>& centroids, const vector<int>& genotype, const vector<CPoint>& points) const {
	vector<int> clusterSizes(maxClusters, 0);
	vector<vector<double>> newCentroidCoordinates(maxClusters);

	#pragma omp parallel for
	for (size_t i = 0; i < points.size(); ++i) {
		int cluster = genotype[i];
		const vector<double>& pointCoordinates = points[i].vGetCoordinates();

		#pragma omp critical
		{
			newCentroidCoordinates[cluster - 1].resize(pointCoordinates.size());
			for (size_t dimension = 0; dimension < pointCoordinates.size(); ++dimension) {
				newCentroidCoordinates[cluster - 1][dimension] += pointCoordinates[dimension];
			}
			++clusterSizes[cluster - 1];
		}
	}

	vector<CPoint> newCentroids(maxClusters);
	for (size_t i = 0; i < maxClusters; ++i) {
		if (clusterSizes[i] > 0) {
			for (size_t dimension = 0; dimension < newCentroidCoordinates[i].size(); ++dimension) {
				newCentroidCoordinates[i][dimension] /= clusterSizes[i];
				newCentroids[i].vAddCoordinate(newCentroidCoordinates[i][dimension]);
			}
		}
		else {
			for (size_t dimension = 0; dimension < centroids[i].vGetCoordinates().size(); ++dimension) {
				newCentroids[i].vAddCoordinate(centroids[i].vGetCoordinates()[dimension]);
			}
		}
	}

	centroids.clear();
	centroids.resize(maxClusters);
	for (size_t i = 0; i < maxClusters; ++i) {
		PointFunctions::assignSourcePointToEmpty(centroids[i], newCentroids[i]);
	}
}