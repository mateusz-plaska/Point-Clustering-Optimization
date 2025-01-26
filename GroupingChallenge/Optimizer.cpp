#include "Optimizer.h"

using namespace NGroupingChallenge;

COptimizer::COptimizer(CGroupingEvaluator& cEvaluator)
	: c_evaluator(cEvaluator)
{
	random_device c_seed_generator;
	c_random_engine.seed(c_seed_generator());
}

void COptimizer::vInitialize()
{
	numeric_limits<double> c_double_limits;
	d_current_best_fitness = c_double_limits.max();

	v_current_best.clear();
	v_current_best.resize(c_evaluator.iGetNumberOfPoints());
}

void COptimizer::vRunIteration()
{
	vector<int> v_candidate(c_evaluator.iGetNumberOfPoints());

	uniform_int_distribution<int> c_candidate_distribution(c_evaluator.iGetLowerBound(), c_evaluator.iGetUpperBound());

	for (size_t i = 0; i < v_candidate.size(); i++)
	{
		v_candidate[i] = c_candidate_distribution(c_random_engine);
	}

	double d_candidate_fitness = c_evaluator.dEvaluate(v_candidate);

	if (d_candidate_fitness < d_current_best_fitness)
	{
		v_current_best = v_candidate;
		d_current_best_fitness = d_candidate_fitness;
	}

	cout << d_current_best_fitness << endl;
}