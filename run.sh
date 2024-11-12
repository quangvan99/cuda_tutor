#!/bin/bash

# Check if source file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <source_file.cpp|source_file.cu>"
    exit 1
fi

# Get source filename from first argument
SOURCE_FILE=$1
EXEC_FILE=${SOURCE_FILE%.*}

# Compile based on file extension
if [[ $SOURCE_FILE == *.cpp ]]; then
    # Compile with g++ for .cpp files
    g++ -o $EXEC_FILE $SOURCE_FILE `pkg-config --cflags --libs opencv`
elif [[ $SOURCE_FILE == *.cu ]]; then
    # Compile with nvcc for .cu files
    nvcc -o $EXEC_FILE $SOURCE_FILE `pkg-config --cflags --libs opencv` -diag-suppress=611
else
    echo "Unsupported file extension. Use .cpp or .cu files only."
    exit 1
fi

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    # Run the program
    ./$EXEC_FILE
    
    # Clean up executable
    rm -rf $EXEC_FILE
else
    echo "Compilation failed."
fi
