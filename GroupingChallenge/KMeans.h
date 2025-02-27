#pragma once

#include "Individual.h"
#include "PointFunctions.h"

class KMeans
{
public:
	KMeans() : maxClusters(0), maxIterations(0) {}
	KMeans(const int& maxClusters, const int& maxIterations)
		: maxClusters(maxClusters), maxIterations(maxIterations) {}

	Individual getIndividual(const vector<CPoint>& points, mt19937& randomEngine);
	Individual getIndividual(const vector<int>& genotypeOriginal, const vector<CPoint>& points);

	void setMaxIterations(const int& maxIterations) {
		this->maxIterations = maxIterations;
	}

private:
	int maxClusters;
	int maxIterations;

	void updateCentroids(vector<CPoint>& centroids, const vector<int>& genotype, const vector<CPoint>& points) const;
};