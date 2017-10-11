cmake_minimum_required(VERSION 2.8)
project( van_pt )
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS}  )

add_executable( van_pt main.cpp Cali_Import.cpp VanPt.cpp)

target_link_libraries( van_pt ${OpenCV_LIBS} 
)


add_definitions(-std=c++11)
