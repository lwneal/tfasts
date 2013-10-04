#pragma once


#include <iostream>
#include <cstdlib>
#include <math.h>
#include <vector>

using namespace std;


// a collection of functions for generating random numbers 
class pRandom
{
public:
	static int randInt() { return rand(); }
	
	// returns a random float between the range min and max
	static float randInRange(float min, float max)
	{
		return min + (max - min) * (float) rand() / (float) RAND_MAX;
	}

	
	// returns a random float from a normal distribution with mean zero
	// and standard deviation sigma (not tested for accuracy, but it seems pretty close).
	static float randGaussian(float sigma)
	{
		float x, y, r2;

		do
		{
		  // choose x, y in uniform square (-1,-1) to (+1,+1) 
		  x = -1 + 2 * randInRange(0, 1);
		  y = -1 + 2 * randInRange(0, 1);

		  // see if it is in the unit circle
		  r2 = x * x + y * y;
		}
		while (r2 > 1.0 || r2 == 0);

		// Box-Muller transform
		return (float)(sigma * y * sqrt (-2.0 * log (r2) / r2));
	}
	
	// return a random integer between 0 and pdf.size() - 1, according
	// to the multinomial distribution described by pdf
	static int randomMultinomial(vector<float>& pdf)
	{
		float r = randInRange(0, 1);
		float sumP = 0;
		for(int i = 0; i < (int)pdf.size(); ++i)
		{
			sumP += pdf[i];
			if( r < sumP ) 
				return i;
		}
		
		return pdf.size() - 1;
	}

	// shuffle the vector of ints v to a new random permutation.
	// i think this is called djikstra's shuffling algorithm.
	static void shuffle(vector<int>& v)
	{
		// loop invariant: the v[0] through v[i] are shuffled
		for(int i = 0; i < (int)v.size(); ++i)
		{
			// chose a random element that is not shuffled
			int r = i + (randInt() % (v.size() - i));
			
			// swap the ith element with the chosen element
			int temp = v[i];
			v[i] = v[r];
			v[r] = temp;
		}
	}
	
	// shuffle a vector<T>
	template <class T>
	static void templatedShuffle(vector<T>& v)
	{
		// loop invariant: the v[0] through v[i] are shuffled
		for(int i = 0; i < v.size(); ++i)
		{
			// chose a random element that is not shuffled
			int r = i + (randInt() % (v.size() - i));
			
			// swap the ith element with the chosen element
			T temp = v[i];
			v[i] = v[r];
			v[r] = temp;
		}
	}
	
	
};
