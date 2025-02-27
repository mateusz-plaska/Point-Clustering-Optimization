#include "Optimizer.h"

using namespace NGroupingChallenge;

COptimizer::COptimizer(CGroupingEvaluator& cEvaluator)
	: c_evaluator(cEvaluator), geneticAlgorithm(c_evaluator.iGetUpperBound(), c_random_engine)
{
	random_device c_seed_generator;
	c_random_engine.seed(c_seed_generator());
}

void COptimizer::vInitialize()
{
	numeric_limits<double> c_double_limits;
	bestFitness = c_double_limits.max();

	v_current_best.clear();

	geneticAlgorithm.initialize(c_evaluator.vGetPoints());
}

void COptimizer::vRunIteration()
{
	geneticAlgorithm.runIteration(c_evaluator.vGetPoints());

	const Individual& bestIndividual = geneticAlgorithm.getBestIndividual();

	if (bestIndividual.fitness < bestFitness)
	{
		v_current_best = *bestIndividual.getGenotype();
		bestFitness = bestIndividual.fitness;
	} 
}