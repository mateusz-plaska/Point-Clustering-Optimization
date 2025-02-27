#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "GroupingEvaluator.h"
#include "GeneticAlgorithm.h"

#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace std;

namespace NGroupingChallenge
{
	class COptimizer
	{
	public:
		COptimizer(CGroupingEvaluator& cEvaluator);

		void vInitialize();
		void vRunIteration();

		vector<int>* pvGetCurrentBest() { return &v_current_best; }

	private:
		CGroupingEvaluator& c_evaluator; 

		vector<int> v_current_best;

		double bestFitness;

		mt19937 c_random_engine;

		GeneticAlgorithm geneticAlgorithm;
	};
}

#endif