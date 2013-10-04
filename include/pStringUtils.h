// a collection of utilities for working with C++ strings
#pragma once

#include <vector>
#include <string>
#include <ctype.h>

using namespace std;

class pStringUtils
{
public:
	
	// like split, but removes empty results
	static void splitNonEmpty(const string s, string delims, vector<string>& outParts)
	{
		vector<string> parts;
		split(s, delims, parts);
		for(int i = 0; i < (int)parts.size(); ++i)
			if( parts[i] != "" ) outParts.push_back(parts[i]);
	}
	
	// split s into a vector of strings by delim and writes the results to outParts (which
	// should be empty before calling this
	static void split(const string s, const char delim, vector<string>& outParts)
	{
		string part = "";
		unsigned int sLen = s.length();
		
		for(unsigned int i = 0; i < sLen; ++i)
		{
			if( s[i] == delim )
			{
				outParts.push_back(part);
				part = "";
			}
			else
				part += s[i];
		}
		
		outParts.push_back(part);
	}
	
	// split with a string specifying a list of delimeters
	static void split(const string s, const string delims, vector<string>& outParts)
	{
		split(replaceChar(s, delims, delims[0]), delims[0], outParts);
	}
	
	// split, then only take the first part
	static string firstPartOfSplit(const string s, const char delim)
	{
		vector<string> parts;
		split(s, delim, parts);
		return parts[0];
	}
	
	static string firstPartOfSplit(const string s, const string delims)
	{
		vector<string> parts;
		split(s, delims, parts);
		return parts[0];
	}
	
	static string secondPartOfSplit(const string s, const string delims)
	{
		vector<string> parts;
		split(s, delims, parts);
		return parts[1];
	}
	
	static string removeChar(string s, char c)
	{
		string r = "";
		for(int i = 0; i < (int)s.size(); ++i)
			if( s[i] != c ) r += s[i];
		return r;
	}
	
	
	static string replaceChar(string s, string charsToReplace, char c)
	{
		for(int i = 0; i < (int)s.size(); ++i)
		{
			for(int j = 0; j < (int)charsToReplace.size(); ++j)
				if( s[i] == charsToReplace[j] )
					s[i] = c;
		}
		return s;
	}

	// Conversion between strings and integers
	static string intToString(int x)
	{
		char buffer[1024];
		sprintf(buffer, "%d", x);
		return string(buffer);
	}
	
	static int stringToInt(string s)
	{
		return atoi( s.c_str() );
	}

	// Conversion between strings and floats.  Precision is the number of
	// digits after the decimal point (-1 is unlimited digits).
	static string floatToString(float x, int precision = -1)
	{
		char buffer[1024];
		sprintf(buffer, "%f", x);
		
		string result(buffer);
		size_t decimalPos = result.find_first_of('.');
		
		if(precision < 0)
			return result;
		else
		{
			if(decimalPos == string::npos)
				return result;
			else
				return result.substr(0, decimalPos + 1 + precision);
		}
	}
	
	static float stringToFloat(string s)
	{
		return (float)atof(s.c_str());
	}

	// Returns true if c is a space, tab character or newline
	static bool isWhitespace(char c)
	{
		return c == ' ' || c == '\n' || c == '\r' || c == '\t';
	}

	// Returns the argument string stripped of leading and trailing whitespace
	// and newline characters. If the string is entirely whitespace or newline
	// characters, the empty string is returned.
	static string pack(const string &val)
	{
		string result = val;
		
		size_t first_char = result.find_first_not_of(" \t\n");
		size_t last_char = result.find_last_not_of(" \t\n");
		
		if(first_char == string::npos || last_char == string::npos) {
			result = "";
		} else {
			result.erase(0, first_char);
			result.erase(last_char - first_char + 1, string::npos);
		}
		return result;
	}
};
