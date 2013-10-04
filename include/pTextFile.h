// this class provides a C++ stream style wrapper on top of C style text file I/O (which is faster than the C++ file IO routines).

#pragma once

#include <cstdio>
#include <cctype>
#include <string>

using namespace std;

// File Mode enumeration
//
// NMODE:			This mode allows neither reading nor writing.
// READ:			Opens a file for reading.  The file must exist.
// WRITE:			Opens a file for writing.  If one with the same
// 					name exists, the contents are erased.
// APPEND:			Writing operations append to the end of the file.
// READ_WRITE:		Opens a file for reading and writing.  The file 
// 					must exist.
// READ_WRITE_NEW:	Creates a new file for reading and writing.  If one
// 					with the same name exists, the contents are erased.
// READ_APPEND:		Opens a file for reading and writing where write
//					operations are appended to the end of the file.
enum PFILE_MODE
{
	PFILE_NMODE,
	PFILE_READ,
	PFILE_WRITE,
	PFILE_APPEND,
	PFILE_READ_WRITE,
	PFILE_READ_WRITE_NEW,
	PFILE_READ_APPEND
};


class pTextFile
{
public:
	// Default constructor
	pTextFile()
	{
		file_ = NULL;
		maxStringChars_ = DEFAULT_MAX_STRING_CHARS;
		fileName_ = "";
		mode_ = PFILE_NMODE;
	}
	
	// Destructor
	~pTextFile() {  close(); }

	// Initialization constructor.  The default mode is PFILE_READ_WRITE.
	pTextFile(const string &fileName, PFILE_MODE mode = PFILE_READ_WRITE)
	{
		file_ = NULL;
		maxStringChars_ = DEFAULT_MAX_STRING_CHARS;
		fileName_ = fileName;
		mode_ = PFILE_NMODE;
		
		open(fileName, mode);
	}

	// Copy constructor.  This creates a new stream to an open file.
	pTextFile(const pTextFile& copy)
	{
		file_ = NULL;
		maxStringChars_ = copy.maxStringChars_;
		fileName_ = copy.fileName_;
		mode_ = copy.mode_;
		if(copy.isOpen())
			open(fileName_, mode_);
	}

	// Assignment operator.  This creates a new file stream if the rhs
	// argument has an open file.
	pTextFile& operator=(const pTextFile& rhs)
	{
		if(this != &rhs)
		{
			close();
			
			maxStringChars_ = rhs.maxStringChars_;
			fileName_ = rhs.fileName_;
			mode_ = rhs.mode_;
			
			if(rhs.isOpen())
				open(fileName_, mode_);
		}
		
		return *this;
	}
	
	// If there is a stream associated with this pFile, close it.  Then,
	// open a new stream under the specified mode.  If the file is opened
	// successfully, the function returns true, otherwise it returns false.
	// The default mode is PFILE_READ_WRITE. 
	bool open(const string &fileName, PFILE_MODE mode = PFILE_READ_WRITE)
	{
		close();
		
		string fileMode;
		
		switch(mode)
		{
			case PFILE_READ:				fileMode = "r";		break;
			case PFILE_WRITE:				fileMode = "w";		break;
			case PFILE_APPEND:				fileMode = "a";		break;
			case PFILE_READ_WRITE:			fileMode = "r+";	break;
			case PFILE_READ_WRITE_NEW:		fileMode = "w+";	break;
			case PFILE_READ_APPEND:			fileMode = "a+";	break;
			default:						file_ = NULL;		return false;
		}
		
		file_ = fopen(fileName.c_str(), fileMode.c_str());
		
		if(isOpen()) 
		{
			mode_ = mode;
			fileName_ = fileName;
			return true;
		} 
		
		return false;		
	}
	
	// Close the stream associated with this pFile.  If the file is closed
	// successfully, true is returned, otherwise the function returns false.
	bool close()
	{
		if(isOpen()) 
		{
			int result = fclose(file_);
			
			file_ = NULL;
			fileName_ = "";
			mode_ = PFILE_NMODE;
			
			return (result == 0);
		} 
		
		return false;
	}
	
	// Returns true if the end of the file represented by this pFile has been
	// reached (or if the file_ member is NULL).  Otherwise, this function
	// returns false
	bool eof() { 
		if(!isOpen()) 
			return true; 
		else 
			return (bool)feof(file_); 
	}
	
	// Returns true if a valid stream exists and false otherwise.
	bool isOpen() const { return (file_ != NULL); }
	
	// Sets the maximum number of characters to read when reading in a string
	void setMaxStringChars(int maxChars) { maxStringChars_ = maxChars; }
	
	// If a file is open, this function closes the file and removes it from 
	// disk. If the operation is successful, the function returns true, 
	// otherwise it returns false.
	bool remove()
	{
		string tmpFileName = fileName_;
		if(close())
			if(::remove(tmpFileName.c_str()) == 0) // the :: means use the remove in the current namespace, not the one defined in this class
				return true;
		
		return false;
	}
	
	// Renames or moves the current open file to a new location and then 
	// reopens the file under the previous mode.  The function returns true
	// upon success and false if an error occurs.
	bool rename(string newName)
	{
		if(isOpen()) 
		{
			PFILE_MODE oldMode = mode_;
			string oldName = fileName_;
			
			if(close())
				if(::rename(oldName.c_str(), newName.c_str()) == 0) 
					if(open(newName, oldMode))
						return true;
		} 
		
		return false;
	}
	
	// Returns the name of the file pointed to by this pFile.
	string name() { return fileName_; }	
	
	// Reads from the file until one of the specified delimiting characters, the end
	// of the file or a new line is reached.  If the first optional boolean 
	// argument is true, the function moves the file pointer ahead to the next 
    // valid character before returning. 
	string readUntil(const char delim, bool advance = true)
	{
		if(!isOpen()) return "";
		
		string result;
		
		char tmp;
		while(true) 
		{
			tmp = getc(file_);
			
			if(tmp != delim && tmp != '\n' && tmp != '\r' && !eof())
				result += tmp;
			else 
			{
				// Advance the file pointer if necessary
				if(advance && !eof()) 
				{
					while(true) 
					{
						tmp = getc(file_);
						if(tmp != delim && tmp !='\n' && tmp != '\r') 
						{
							ungetc(tmp, file_);
							break;
						}
						if(eof())
							break;
					}
				} 
				else if(!eof()) 
				{
					if(tmp == delim)
						ungetc(tmp, file_);
				}
				break;
			}
		}
		
		return result;
	}

	// Reads until a non-whitespace or endline character is reached, or until
	// the end of the file.  The whitespace is not returned.
	void skipSpace()
	{
		if(!isOpen()) return;
		
		char tmp;
		while(true) 
		{
			if(eof())
				break;
			
			tmp = getc(file_);
			if(tmp != ' ' && tmp != '\n' && tmp != '\t' && tmp != '\r') 
			{
				ungetc(tmp, file_);
				break;
			}
		}		
	}

	// Reads from the current position to the end of the line or until the end
    // of the file is reached.
	string readLine() { return readUntil('\0', true); }

	/// output stream operators ///
	
	friend pTextFile& operator<<( pTextFile &out, const int& val)					{ fprintf(out.file_, "%d", val);	return out; }
	friend pTextFile& operator<<( pTextFile &out, const short int& val)				{ fprintf(out.file_, "%hd", val);	return out; }
	friend pTextFile& operator<<( pTextFile &out, const long int& val)				{ fprintf(out.file_, "%ld", val);	return out; }
	friend pTextFile& operator<<( pTextFile &out, const unsigned int& val)			{ fprintf(out.file_, "%u", val);	return out; }
	friend pTextFile& operator<<( pTextFile &out, const unsigned short int& val)	{ fprintf(out.file_, "%hu", val);	return out; }
	friend pTextFile& operator<<( pTextFile &out, const unsigned long int& val)		{ fprintf(out.file_, "%lu", val);	return out; }
	friend pTextFile& operator<<( pTextFile &out, const float& val)					{ fprintf(out.file_, "%f", val);	return out; }
	friend pTextFile& operator<<( pTextFile &out, const double& val)				{ fprintf(out.file_, "%lf", val);	return out; }
	friend pTextFile& operator<<( pTextFile &out, const long double& val)			{ fprintf(out.file_, "%Lf", val);	return out; }
	friend pTextFile& operator<<( pTextFile &out, const bool& val)					{ fprintf(out.file_, "%d", val);	return out; }
	friend pTextFile& operator<<( pTextFile &out, const char& val)					{ fprintf(out.file_, "%c", val);	return out; }
	friend pTextFile& operator<<( pTextFile &out, const unsigned char& val)			{ fprintf(out.file_, "%c", val);	return out; }
	friend pTextFile& operator<<( pTextFile &out, const char* val)					{ fprintf(out.file_, "%s", val);	return out; }
	friend pTextFile& operator<<( pTextFile &out, const string& val)				{ return operator<<(out, val.c_str()); }
	friend pTextFile& operator<<( pTextFile &out, pTextFile& (*manip)(pTextFile&))	{ return manip(out); }

	// Output manipulators
	friend pTextFile& endl(pTextFile& out)	{ fprintf(out.file_, "%c", '\n'); return out; }
	friend pTextFile& tab(pTextFile& out)	{ fprintf(out.file_, "%c", '\t'); return out; }
	
private:
	// Internal file pointer.  This is NULL whenever the pFile is
	// invalid.
	FILE* file_;
	
	// Maximum number of characters to read when reading in a string
	int maxStringChars_;
	static const int DEFAULT_MAX_STRING_CHARS = 80;
	
	// The fileName of this pTextFile.  The string
	// is empty when no file is open.
	string fileName_;
	
	// The current I/O mode this pFile is in.  The default value is
	// NMODE.
	PFILE_MODE mode_;
};
