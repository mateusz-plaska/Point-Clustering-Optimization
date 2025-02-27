#pragma once

#include "Point.h"
using namespace NGroupingChallenge;

class PointFunctions
{
public:
	static void assignSourcePointToEmpty(CPoint& target, const CPoint& source) {
		for (size_t i = 0; i < source.vGetCoordinates().size(); ++i) {
			target.vAddCoordinate(source.vGetCoordinates()[i]);
		}
	}

	static double calculateDistance(const CPoint& source, const CPoint& target) {
		double squaredDistance = 0;

		for (size_t i = 0; i < source.vGetCoordinates().size(); i++)
		{
			double diff = source.vGetCoordinates()[i] - target.vGetCoordinates()[i];
			squaredDistance += ((diff) * (diff));
		}

		return squaredDistance;
	}
};
