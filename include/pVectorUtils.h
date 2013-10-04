#pragma once

#include <iostream>
#include <vector>
#include <limits.h>
#include <float.h>
#include <assert.h>
#include <math.h>

using namespace std;


class pVectorUtils
{
public:	
	static bool sort_int_bool_pair(const pair<int,float>& a, const pair<int,float>& b)
	{
		return a.second > b.second;
	}

	static double log2( double n )  
	{  
		// log(n)/log(2) is log2.  
		return log( n ) / log( 2.0 );  
	}
	// various versions of argmin and argmax
	static int argmax(vector<int>& v)
	{
		int maxSoFar = INT_MIN;
		int maxIndex = 0;
		for(int i = 0; i < (int)v.size(); ++i)
		{
			if( v[i] > maxSoFar )
			{
				maxSoFar = v[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	static int argmax(vector<float>& v)
	{
		float maxSoFar = -FLT_MIN;
		int maxIndex = 0;
		for(int i = 0; i < (int)v.size(); ++i)
		{
			if( v[i] > maxSoFar )
			{
				maxSoFar = v[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	static int argmin(vector<int>& v)
	{
		int minSoFar = INT_MAX;
		int minIndex = 0;
		for(int i = 0; i < (int)v.size(); ++i)
		{
			if( v[i] < minSoFar )
			{
				minSoFar = v[i];
				minIndex = i;
			}
		}
		return minIndex;
	}
	
	static int argmin(vector<float>& v)
	{
		float minSoFar = FLT_MAX;
		int minIndex = 0;
		for(int i = 0; i < (int)v.size(); ++i)
		{
			if( v[i] < minSoFar )
			{
				minSoFar = v[i];
				minIndex = i;
			}
		}
		return minIndex;
	}
	
	// if v contains x, returns the index of x, otherwise -1
	template <class T> static int index_of(vector<T>& v, const T& x)
	{
		for(int i = 0; i < v.size(); ++i)
			if( v[i] == x ) return i;
		return -1;
	}
	
	template <class T> static bool vector_contains(vector<T>& v, const T& x)
	{
		for(int i = 0; i < v.size(); ++i)
			if( v[i] == x ) return true;
		return false;
	}
	
	// normalize a vector so its elements add up to 1
	static void normalizeVectorToPDF(vector<float>& w)
	{
		float sum = 0;
		for(int i = 0; i < (int)w.size(); ++i) 
			sum += w[i];
		for(int i = 0; i < (int)w.size(); ++i) 
			w[i] /= sum;
	}
	
	static float giniForDistribution(vector<float>& prob)
	{
		float gini = 1;
		for(int i = 0; i < (int)prob.size(); ++i) gini -= prob[i] * prob[i];
		return gini;
	}
	
	template <class T> static void cout_vector(vector<T>& v)
	{
		for(int i = 0; i < (int)v.size(); ++i) cout << v[i] << (i == v.size() - 1 ? "" : "\n");
		cout << endl;
	}
	
	static void append(vector<float>& src, vector<float>& dst)
	{
		for(int i = 0; i < (int)src.size(); ++i) dst.push_back(src[i]);
	}
};

// this is all commented out because i am getting some seriously fucked up linker errors (for example, it claims that a different symbol is duplicated when compiling for debug and release...) also 
// all the code that the errors are coming from is in use in a ton of other projects with no problem. moving everything into static member functions of a pVectorUtils
// class fixed it, but im not sure why it broke in the first place

/*
// this collection of functions is just some syntactic sugar for pushing more than one thing to a vector at once. unfortunately, it could not
// be done with varargs. this file defines cases for 1 - 10 arguments. example use: vector_multi_push<int>(someInts, 1, 2, 3, 4, 5);
template <class T> void vector_multi_push(vector<T>& v, const T& x1)																															{ v.push_back(x1); }
template <class T> void vector_multi_push(vector<T>& v, const T& x1, const T& x2)																												{ v.push_back(x1); v.push_back(x2); }
template <class T> void vector_multi_push(vector<T>& v, const T& x1, const T& x2, const T& x3)																									{ v.push_back(x1); v.push_back(x2); v.push_back(x3); }
template <class T> void vector_multi_push(vector<T>& v, const T& x1, const T& x2, const T& x3, const T& x4)																						{ v.push_back(x1); v.push_back(x2); v.push_back(x3); v.push_back(x4); }
template <class T> void vector_multi_push(vector<T>& v, const T& x1, const T& x2, const T& x3, const T& x4, const T& x5)																		{ v.push_back(x1); v.push_back(x2); v.push_back(x3); v.push_back(x4); v.push_back(x5); }
template <class T> void vector_multi_push(vector<T>& v, const T& x1, const T& x2, const T& x3, const T& x4, const T& x5, const T& x6)															{ v.push_back(x1); v.push_back(x2); v.push_back(x3); v.push_back(x4); v.push_back(x5); v.push_back(x6); }
template <class T> void vector_multi_push(vector<T>& v, const T& x1, const T& x2, const T& x3, const T& x4, const T& x5, const T& x6, const T& x7)												{ v.push_back(x1); v.push_back(x2); v.push_back(x3); v.push_back(x4); v.push_back(x5); v.push_back(x6); v.push_back(x7); }
template <class T> void vector_multi_push(vector<T>& v, const T& x1, const T& x2, const T& x3, const T& x4, const T& x5, const T& x6, const T& x7, const T& x8)									{ v.push_back(x1); v.push_back(x2); v.push_back(x3); v.push_back(x4); v.push_back(x5); v.push_back(x6); v.push_back(x7); v.push_back(x8); }
template <class T> void vector_multi_push(vector<T>& v, const T& x1, const T& x2, const T& x3, const T& x4, const T& x5, const T& x6, const T& x7, const T& x8, const T& x9)					{ v.push_back(x1); v.push_back(x2); v.push_back(x3); v.push_back(x4); v.push_back(x5); v.push_back(x6); v.push_back(x7); v.push_back(x8); v.push_back(x9); }
template <class T> void vector_multi_push(vector<T>& v, const T& x1, const T& x2, const T& x3, const T& x4, const T& x5, const T& x6, const T& x7, const T& x8, const T& x9, const T& x10)		{ v.push_back(x1); v.push_back(x2); v.push_back(x3); v.push_back(x4); v.push_back(x5); v.push_back(x6); v.push_back(x7); v.push_back(x8); v.push_back(x9); v.push_back(x10); }

// append the vector src to the end of dst
void appendVector(vector<float>& src, vector<float>& dst)
{
	for(int i = 0; i < src.size(); ++i) dst.push_back(src[i]);
}

// copy src to dst (assuming dst is empty). if dst is non-empty, this appends
void copy2DVector(vector< vector<float> >& src, vector< vector<float> >& dst)
{
	for(int i = 0; i < src.size(); ++i)
		dst.push_back(src[i]);
}


// if v contains x, returns the index of x, otherwise -1
template <class T> int index_of(vector<T>& v, const T& x)
{
	for(int i = 0; i < v.size(); ++i)
		if( v[i] == x ) return i;
	return -1;
}

template <class T> void cout_vector(vector<T>& v)
{
	for(int i = 0; i < v.size(); ++i) cout << v[i] << (i == v.size() - 1 ? "" : ",");
	cout << endl;
}

// push n copies of val into the vector
template <class T> void vector_push_n_of(vector<T>& v, const T& val, int n)
{
	for(int i = 0; i < n; ++i) v.push_back(val);
}

float max_vector(vector<float>& v)
{
	float x = FLT_MIN;
	for(int i = 0; i < v.size(); ++i)
		x = max(x, v[i]);
	return x;
}

float min_vector(vector<float>& v)
{
	float x = FLT_MAX;
	for(int i = 0; i < v.size(); ++i)
		x = min(x, v[i]);
	return x;
}

// various versions of argmin and argmax
int argmax(vector<int> v)
{
	int maxSoFar = INT_MIN;
	int maxIndex = 0;
	for(int i = 0; i < v.size(); ++i)
	{
		if( v[i] > maxSoFar )
		{
			maxSoFar = v[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}

int argmax(vector<float> v)
{
	float maxSoFar = FLT_MIN;
	int maxIndex = 0;
	for(int i = 0; i < v.size(); ++i)
	{
		if( v[i] > maxSoFar )
		{
			maxSoFar = v[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}

int argmin(vector<int> v)
{
	int minSoFar = INT_MAX;
	int minIndex = 0;
	for(int i = 0; i < v.size(); ++i)
	{
		if( v[i] < minSoFar )
		{
			minSoFar = v[i];
			minIndex = i;
		}
	}
	return minIndex;
}

int argmin(vector<float> v)
{
	float minSoFar = FLT_MAX;
	int minIndex = 0;
	for(int i = 0; i < v.size(); ++i)
	{
		if( v[i] < minSoFar )
		{
			minSoFar = v[i];
			minIndex = i;
		}
	}
	return minIndex;
}

// make a uniform vector with n elements
void makeUniformDistribution(vector<float>& w, int n)
{
	for(int i = 0; i < n; ++i) w.push_back(1.0/(float)n);
}

// normalize a vector so its elements add up to 1
void normalizeVectorToPDF(vector<float>& w)
{
	float sum = 0;
	for(int i = 0; i < w.size(); ++i) sum += w[i];
	assert(sum != 0);
	for(int i = 0; i < w.size(); ++i) w[i] /= sum;
}
*/
