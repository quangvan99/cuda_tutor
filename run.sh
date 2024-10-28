#!/bin/bash

# Kiểm tra xem đã nhập tên file chưa
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <source_file.cpp|source_file.cu>"
    exit 1
fi

# Lấy tên file nguồn từ tham số đầu tiên
SOURCE_FILE=$1
EXEC_FILE=${SOURCE_FILE%.*}

# Kiểm tra đuôi file và biên dịch dựa trên loại file
if [[ $SOURCE_FILE == *.cpp ]]; then
    # Compile với g++ nếu là file .cpp
    g++ -o $EXEC_FILE $SOURCE_FILE `pkg-config --cflags --libs opencv`
elif [[ $SOURCE_FILE == *.cu ]]; then
    # Compile với nvcc nếu là file .cu
    nvcc -o $EXEC_FILE $SOURCE_FILE `pkg-config --cflags --libs opencv` -diag-suppress=611
else
    echo "Unsupported file extension. Use .cpp or .cu files only."
    exit 1
fi

# Kiểm tra xem quá trình biên dịch có thành công không
if [ $? -eq 0 ]; then
    # Chạy chương trình
    ./$EXEC_FILE

    # Xoá file thực thi sau khi chạy
    rm -rf $EXEC_FILE
else
    echo "Compilation failed."
fi
